"""Candidate generation (blocking).

Retrieval: every record becomes a sparse TF-IDF vector over discrete *keys*
built from its normalised fields — core-name tokens, phonetic-skeleton tokens
(transliteration robust: Lakshmi/Laxmi, Hindi script -> same skeleton), name
bigrams, address words, house numbers, postal codes and address bigrams.
Keys shared by more than `key_max_df` of the bucket (e.g. "pvt", "road") are
dropped, so posting lists stay short. For every Source-1 record the exact
top-K cosine neighbours in Source 2 and Source 3 are retrieved inside the same
country bucket with a multi-threaded sparse top-n product (sparse_dot_topn):
cost is O(N * K * avg posting length), not O(N^2).

Scoring: every retrieved pair gets exact character n-gram TF-IDF cosines of
the name, the name skeleton and the address (hashed n-grams, IDF over the
bucket); the list is pruned per S1 record and source with a cheap blended
score. The pruned list is what the stage-1 filter and the matcher score.
"""
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from prep import col

# exact cosine views: name -> (normalised column, char n-gram range)
VIEWS = {"name": ("n_core", (2, 4)), "skel": ("n_skel", (2, 3)), "addr": ("a_full", (3, 4))}
KEY_FIELDS = ("n_core", "n_skel", "a_full", "a_words", "a_nums", "a_postal")


def _num_variants(n):
    """House-number keys robust to a dropped first/last digit (9224 ~ 224 ~ 922)."""
    out = ["#" + n]
    if len(n) >= 3 and n.isdigit():
        out += ["#" + n[1:].lstrip("0"), "#" + n[:-1]]
    return out


def key_doc(core, skel, full, words, nums, postal, variants=False):
    t = core.split()
    a = full.split()
    nk = ([k for x in nums.split() for k in _num_variants(x)] if variants
          else ["#" + x for x in nums.split()])
    return (t + ["~" + x for x in skel.split()] + [x + "_" + y for x, y in zip(t, t[1:])]
            + ["a:" + x for x in words.split()] + nk
            + ["p:" + x for x in postal.split()] + ["b:" + x + "_" + y for x, y in zip(a, a[1:])])


def _identity(x):
    return x


def _hash_worker(args):
    """Runs in a worker process: raw term counts of one chunk of records."""
    kind, cols, ngr, nf = args
    if kind in ("key", "keyv"):
        docs = [key_doc(*r, variants=kind == "keyv") for r in zip(*cols)]
        hv = HashingVectorizer(analyzer=_identity, n_features=nf, alternate_sign=False,
                               norm=None, dtype=np.float32)
    elif kind == "tok":
        docs = [s.split() for s in cols[0]]
        hv = HashingVectorizer(analyzer=_identity, n_features=nf, alternate_sign=False,
                               norm=None, binary=True, dtype=np.float32)
    else:
        docs = cols[0]
        hv = HashingVectorizer(analyzer="char_wb", ngram_range=ngr, n_features=nf,
                               alternate_sign=False, norm=None, lowercase=False,
                               dtype=np.float32)
    return hv.transform(docs).tocsr()


class HashedTfidf:
    """Sublinear-TF x IDF, L2-normalised, over hashed features; parallel.

    IDF is fitted on all records of one country bucket (S1+S2+S3, no labels).
    kind: "char" (char_wb n-grams of one column), "key" (blocking keys) or
    "tok" (binary whitespace tokens, used for set-overlap features).
    """

    def __init__(self, pool, kind, cols, ngr=None, nf=2 ** 20, max_df=None, chunk=50000):
        self.pool, self.kind, self.cols, self.ngr, self.nf = pool, kind, cols, ngr, nf
        self.max_df, self.chunk = max_df, chunk

    def _counts(self, table, rows=None):
        data = [col(table, c, rows) for c in self.cols]
        n = len(data[0])
        jobs = ((self.kind, [d[s:s + self.chunk] for d in data], self.ngr, self.nf)
                for s in range(0, n, self.chunk))
        return self.pool.imap(_hash_worker, jobs)

    def fit(self, tables):
        df = np.zeros(self.nf, dtype=np.int64)
        n = 0
        for t in tables:
            for m in self._counts(t):
                df += np.bincount(m.indices, minlength=self.nf)
                n += m.shape[0]
        self.df, self.n = df, n
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        if self.max_df is not None:
            self.idf[df > self.max_df * n] = 0.0
        return self

    def transform(self, table, rows=None, normalise=True):
        parts = list(self._counts(table, rows))
        X = sparse.vstack(parts).tocsr() if parts else sparse.csr_matrix((0, self.nf), dtype=np.float32)
        if self.kind == "tok":
            return X
        X.data = (1 + np.log(X.data)) * self.idf[X.indices]
        X.eliminate_zeros()
        if normalise:
            nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            X = sparse.diags(1 / np.maximum(nrm, 1e-12)).astype(np.float32) @ X
        return X.tocsr()


def topk_sparse(A, B, k, threads=None):
    """Exact top-k cosine neighbours (rows of B) of each row of A: (i, j, score)."""
    from sparse_dot_topn import sp_matmul_topn
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=min(k, B.shape[0]),
                       n_threads=threads or os.cpu_count()).tocsr()
    i = np.repeat(np.arange(A.shape[0]), np.diff(C.indptr))
    return i, C.indices.astype(np.int64), C.data.astype(np.float32)


def rowwise_cos(A, B, ia, ib, chunk=500000):
    res = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        res[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(axis=1)).ravel()
    return res


def pair_cos(vec, L, R, i1, i2, XL=None, block=400000):
    """Exact cosine of (L[i1], R[i2]); R is transformed block-wise to bound memory."""
    XL = vec.transform(L) if XL is None else XL
    out = np.zeros(len(i1), dtype=np.float32)
    order = np.argsort(i2, kind="stable")
    bounds = np.searchsorted(i2[order], np.arange(0, R.num_rows + block, block))
    for b, s in enumerate(range(0, R.num_rows, block)):
        sel = order[bounds[b]:bounds[b + 1]]
        if len(sel) == 0:
            continue
        XR = vec.transform(R, np.arange(s, min(s + block, R.num_rows)))
        out[sel] = rowwise_cos(XL, XR, i1[sel], i2[sel] - s)
    return out


def generate_candidates(L, targets, pool, k_key=30, key_max_df=0.0005, keep_per_source=6,
                        min_score=0.25, prune_score="cheap", num_variants=False, log=print):
    """Blocked + pruned candidate pairs for one country bucket.

    L: S1 records of the bucket; targets: {"S2": table, "S3": table} of the same bucket.
    Returns {tag: DataFrame(i1_local, i2_local, src, cos_key, cos_*, cheap, cheap_rank)}.
    """
    tables = [L] + list(targets.values())
    kv = HashedTfidf(pool, "keyv" if num_variants else "key", KEY_FIELDS, nf=2 ** 24,
                     max_df=key_max_df).fit(tables)
    AL = kv.transform(L)
    out = {}
    for tag, R in targets.items():
        i, j, s = topk_sparse(AL, kv.transform(R), k_key)
        out[tag] = pd.DataFrame({"i1_local": i, "i2_local": j, "src": tag, "cos_key": s})
        log(f"    retrieved {tag}: {len(i)} pairs")
    del AL, kv
    for view, (c, ngr) in VIEWS.items():
        vec = HashedTfidf(pool, "char", (c,), ngr=ngr).fit(tables)
        XL = vec.transform(L)
        for tag, R in targets.items():
            d = out[tag]
            d[f"cos_{view}"] = pair_cos(vec, L, R, d["i1_local"].values, d["i2_local"].values, XL)
        del XL, vec
    log("    exact cosines done")
    for tag, d in out.items():
        nm = np.maximum(d["cos_name"], d["cos_skel"])
        d["cheap"] = (0.65 * nm + 0.35 * d["cos_addr"]).astype(np.float32)
        # "blend" also trusts the IDF-weighted key match (rare address/name keys)
        d["blend"] = (0.5 * d["cheap"] + 0.5 * d["cos_key"]).astype(np.float32)
        out[tag] = prune(d, keep_per_source, min_score, prune_score)
    return out


def prune(cand, keep_per_source, min_score, score="cheap"):
    cand = cand[cand[score] >= min_score]
    cand = cand.sort_values(["i1_local", score], ascending=[True, False], kind="stable")
    cand["cheap_rank"] = cand.groupby("i1_local").cumcount()
    return cand[cand["cheap_rank"] < keep_per_source].reset_index(drop=True)


def cheap_features(cand, k1="i1_local", k2="i2_local"):
    """Vectorised features for the stage-1 candidate filter (no string ops).

    cand holds a single target source, so the (i1, src) / (src, i2) groups of
    the original formulation reduce to i1 / i2 groups.
    """
    f = cand[["cos_name", "cos_skel", "cos_addr", "cheap", "cheap_rank"]].copy()
    f["is_s3"] = (cand["src"] == "S3").astype(int)
    g1 = f.groupby(cand[k1].values)
    g2 = f.groupby(cand[k2].values)
    for c in ("cos_name", "cos_skel", "cos_addr", "cheap"):
        f[c + "_gap1"] = f[c] - g1[c].transform("max")
        f[c + "_gap2"] = f[c] - g2[c].transform("max")
    f["n_s1_for_cand"] = g2["cheap"].transform("size")
    return f
