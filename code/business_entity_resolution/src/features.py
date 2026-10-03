"""Pairwise features for (Source-1 record, candidate) pairs.

All features are country-agnostic similarity/context signals, so the model
transfers to countries not seen in training (France in the test set).

Everything is vectorised: fuzzy string scores use rapidfuzz.process.cpdist
(C++, multi-threaded, element-wise over pair lists) and the set-overlap
features (Jaccard, IDF-weighted overlap, house numbers, postal codes) are
row-wise products of sparse binary token matrices.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from prep import col

TOKEN_FIELDS = ("n_core", "a_words", "a_nums", "a_postal", "n_skel")


class BucketTokens:
    """Binary token matrices (+ IDF) for every source of one country bucket.

    Tokens are hashed (2^22 buckets, collisions negligible at this vocabulary
    size); IDF is computed over all records of the bucket (S1, S2, S3), no labels.
    """

    def __init__(self, pool, frames):
        from blocking import HashedTfidf
        self.mats = {tag: {} for tag in frames}
        self.idf = {}
        for fld in TOKEN_FIELDS:
            hv = HashedTfidf(pool, "tok", (fld,), nf=2 ** 22)
            dfreq, n = np.zeros(hv.nf, dtype=np.float64), 0
            for tag, t in frames.items():
                m = hv.transform(t)
                m.data[:] = 1.0
                self.mats[tag][fld] = m
                dfreq += np.bincount(m.indices, minlength=hv.nf)
                n += m.shape[0]
            self.idf[fld] = (np.log((n + 1) / (dfreq + 1)) + 1).astype(np.float32)


def _row_inter(A, B, ia, ib, w=None, chunk=500000):
    """|A_i ∩ B_j| (or IDF-weighted) for each pair (ia, ib) of binary rows."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        m = A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).tocsr()
        out[s:s + chunk] = (m @ w) if w is not None else np.asarray(m.sum(axis=1)).ravel()
    return out


def _pair_scores(scorer, a, b, workers):
    return cpdist(a, b, scorer=scorer, workers=workers).astype(np.float32)


def base_features(cand, L, R, tokL, tokR, idf, workers=-1, chunk=1000000, feature_set="v1"):
    """Pair features that depend only on the two records (not on competitors).

    cand: DataFrame with i1_local (row in L), i2_local (row in R), cos_*/cheap.
    L, R: normalised arrow tables of the S1 bucket and the target-source bucket.
    tokL, tokR: {field: binary token matrix} for L and R; idf: {field: vector}.
    Pairs are processed in chunks to bound memory.
    """
    parts = [_base_features(cand.iloc[s:s + chunk], L, R, tokL, tokR, idf, workers, feature_set)
             for s in range(0, len(cand), chunk)]
    return (pd.concat(parts, ignore_index=True) if parts
            else _base_features(cand, L, R, tokL, tokR, idf, workers, feature_set))


def _num_split(v):
    """'849a' -> (849, 'a'); '' -> (nan, '')."""
    d = "".join(ch for ch in v if ch.isdigit())
    return (float(d) if d else np.nan), v[len(d):] if v[:len(d)] == d else ""


def _v2_features(f, a, b, fa, fb, L, R, i1, i2, tokL, tokR, idf, workers):
    """Extra signals found in the error analysis (country-agnostic)."""
    n = len(a)
    # house-number relations: true matches drop a digit / add a letter suffix,
    # different businesses on the same street sit at *nearby* numbers
    both = (fa != "") & (fb != "")
    f["a_num_lev"] = np.where(both, _pair_scores(Levenshtein.normalized_similarity, fa, fb, workers),
                              np.nan).astype(np.float32)
    dist = cpdist(fa, fb, scorer=Levenshtein.distance, workers=workers)
    lena = np.fromiter((len(x) for x in fa), np.int32, n)
    lenb = np.fromiter((len(x) for x in fb), np.int32, n)
    f["a_num_drop"] = np.where(both, (dist == 1) & (np.abs(lena - lenb) == 1), np.nan).astype(np.float32)
    na_ = np.array([_num_split(x)[0] for x in fa]) if n else np.zeros(0)
    nb_ = np.array([_num_split(x)[0] for x in fb]) if n else np.zeros(0)
    f["a_num_core_eq"] = np.where(both, na_ == nb_, np.nan).astype(np.float32)
    f["a_num_logdiff"] = np.where(both, np.log1p(np.abs(na_ - nb_)), np.nan).astype(np.float32)
    # names: spacing-insensitive ratio (web domains, glued words)
    f["n_nospace_ratio"] = _pair_scores(fuzz.ratio, np.array([x.replace(" ", "") for x in a], object),
                                        np.array([x.replace(" ", "") for x in b], object), workers)
    # skeleton token overlap (transliteration robust)
    A, B = tokL["n_skel"], tokR["n_skel"]
    na, nb = np.asarray(A.sum(1)).ravel()[i1], np.asarray(B.sum(1)).ravel()[i2]
    inter = _row_inter(A, B, i1, i2)
    f["n_skel_jacc"] = np.where((na == 0) & (nb == 0), np.nan,
                                inter / np.maximum(1, na + nb - inter)).astype(np.float32)
    # name rarity: coined brand names (unique tokens) vs generic words
    w = idf["n_core"]
    A, B = tokL["n_core"], tokR["n_core"]
    ca, cb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    f["n_idf_mean_l"] = ((A @ w) / np.maximum(1, ca))[i1].astype(np.float32)
    f["n_idf_mean_r"] = ((B @ w) / np.maximum(1, cb))[i2].astype(np.float32)
    f["n_script"] = (col(L, "n_script", i1).astype(np.float32)
                     + col(R, "n_script", i2).astype(np.float32))


def _base_features(cand, L, R, tokL, tokR, idf, workers, feature_set="v1"):
    i1, i2 = cand["i1_local"].values, cand["i2_local"].values
    n = len(cand)
    f = {}
    for c in ("cos_name", "cos_skel", "cos_addr", "cheap"):
        f[c] = cand[c].values.astype(np.float32)
    f["cheap_rank"] = cand["cheap_rank"].values.astype(np.float32)
    f["is_s3"] = np.full(n, float(cand["src"].iat[0] == "S3") if n else 0.0, np.float32)

    g = lambda t, name, idx: col(t, name, idx)  # noqa: E731
    a, b = g(L, "n_core", i1), g(R, "n_core", i2)
    f["n_ratio"] = _pair_scores(fuzz.ratio, a, b, workers)
    f["n_partial"] = _pair_scores(fuzz.partial_ratio, a, b, workers)
    f["n_tsort"] = _pair_scores(fuzz.token_sort_ratio, a, b, workers)
    f["n_tset"] = _pair_scores(fuzz.token_set_ratio, a, b, workers)
    f["n_jw"] = _pair_scores(JaroWinkler.similarity, a, b, workers)
    f["n_skel_ratio"] = _pair_scores(fuzz.token_sort_ratio, g(L, "n_skel", i1),
                                     g(R, "n_skel", i2), workers)
    f["n_full_tset"] = _pair_scores(fuzz.token_set_ratio, g(L, "n_full", i1),
                                    g(R, "n_full", i2), workers)

    # token-set overlaps on the core name
    A, B, w = tokL["n_core"], tokR["n_core"], idf["n_core"]
    na, nb = np.asarray(A.sum(1)).ravel()[i1], np.asarray(B.sum(1)).ravel()[i2]
    wa, wb = (A @ w)[i1], (B @ w)[i2]
    inter = _row_inter(A, B, i1, i2)
    winter = _row_inter(A, B, i1, i2, w)
    union = na + nb - inter
    f["n_jacc"] = np.where((na == 0) & (nb == 0), np.nan, inter / np.maximum(1, union))
    both = (na > 0) & (nb > 0)
    f["n_idf_min"] = np.where(both, winter / np.maximum(1e-9, np.minimum(wa, wb)), np.nan)
    f["n_idf_jacc"] = np.where(both, winter / np.maximum(1e-9, wa + wb - winter), np.nan)
    f["n_words_min"] = np.minimum(na, nb)

    f["n_first_eq"] = (g(L, "n_first", i1) == g(R, "n_first", i2)).astype(np.float32)
    f["n_exact"] = (a == b).astype(np.float32)
    la_, ra_ = g(L, "n_acro", i1), g(R, "n_acro", i2)
    f["n_acro"] = np.fromiter(
        (bool(p and p == y.replace(" ", "")) or bool(q and q == x.replace(" ", ""))
         for p, q, x, y in zip(la_, ra_, a, b)), np.float32, n)
    lena = np.fromiter((len(x) for x in a), np.float32, n)
    lenb = np.fromiter((len(x) for x in b), np.float32, n)
    f["n_len_diff"] = np.abs(lena - lenb) / np.maximum(1, np.maximum(lena, lenb))
    f["n_containment"] = np.fromiter(
        (bool(x) and bool(y) and (x in y or y in x) for x, y in zip(a, b)), np.float32, n)

    # ---- address
    x, y = g(L, "a_full", i1), g(R, "a_full", i2)
    ex = np.fromiter((not s for s in x), bool, n)
    ey = np.fromiter((not s for s in y), bool, n)
    ok = ~ex & ~ey
    f["a_empty"] = ex.astype(np.float32) + ey.astype(np.float32)
    nanf = lambda v: np.where(ok, v, np.nan).astype(np.float32)  # noqa: E731
    f["a_ratio"] = nanf(_pair_scores(fuzz.ratio, x, y, workers))
    f["a_tset"] = nanf(_pair_scores(fuzz.token_set_ratio, x, y, workers))
    f["a_partial"] = nanf(_pair_scores(fuzz.partial_ratio, x, y, workers))

    A, B, w = tokL["a_words"], tokR["a_words"], idf["a_words"]
    na, nb = np.asarray(A.sum(1)).ravel()[i1], np.asarray(B.sum(1)).ravel()[i2]
    wa, wb = (A @ w)[i1], (B @ w)[i2]
    inter = _row_inter(A, B, i1, i2)
    winter = _row_inter(A, B, i1, i2, w)
    f["a_jacc"] = nanf(np.where((na == 0) & (nb == 0), np.nan,
                                inter / np.maximum(1, na + nb - inter)))
    both = (na > 0) & (nb > 0)
    f["a_idf_min"] = nanf(np.where(both, winter / np.maximum(1e-9, np.minimum(wa, wb)), np.nan))
    f["a_idf_jacc"] = nanf(np.where(both, winter / np.maximum(1e-9, wa + wb - winter), np.nan))

    A, B = tokL["a_nums"], tokR["a_nums"]
    na, nb = np.asarray(A.sum(1)).ravel()[i1], np.asarray(B.sum(1)).ravel()[i2]
    inter = _row_inter(A, B, i1, i2)
    f["a_num_jacc"] = nanf(np.where((na == 0) & (nb == 0), np.nan,
                                    inter / np.maximum(1, na + nb - inter)))
    f["a_num_conflict"] = nanf(((na > 0) & (nb > 0) & (inter == 0)).astype(np.float32))
    fa, fb = g(L, "a_first_num", i1), g(R, "a_first_num", i2)
    f["a_first_num_eq"] = nanf(np.where((fa != "") & (fb != ""), (fa == fb), np.nan))
    A, B = tokL["a_postal"], tokR["a_postal"]
    na, nb = np.asarray(A.sum(1)).ravel()[i1], np.asarray(B.sum(1)).ravel()[i2]
    inter = _row_inter(A, B, i1, i2)
    f["a_postal"] = nanf(np.where((na > 0) & (nb > 0), inter > 0, np.nan))
    f["a_tail_eq"] = nanf(g(L, "a_tail", i1) == g(R, "a_tail", i2))
    alen_a = np.fromiter((len(s) for s in x), np.float32, n)
    alen_b = np.fromiter((len(s) for s in y), np.float32, n)
    f["a_len_min"] = nanf(np.minimum(alen_a, alen_b))
    f["a_landmark"] = (g(L, "a_landmark", i1).astype(np.float32)
                       + g(R, "a_landmark", i2).astype(np.float32))
    if feature_set == "v2":
        _v2_features(f, a, b, fa, fb, L, R, i1, i2, tokL, tokR, idf, workers)
    return pd.DataFrame(f)


def context_features(f, i1, src, i2):
    """How each pair ranks against its competitors (same S1+source / same candidate)."""
    f = f.copy()
    key1 = [i1, src]
    g = f.groupby(key1)
    for c in ("cos_name", "n_tset", "cos_addr", "cheap"):
        f[f"{c}_gap1"] = f[c] - g[c].transform("max")
        f[f"{c}_rank1"] = g[c].rank(ascending=False, method="min")
    f["n_cands_src"] = g["cheap"].transform("size")
    key2 = [src, i2]
    g2 = f.groupby(key2)
    for c in ("cheap", "n_tset"):
        f[f"{c}_gap2"] = f[c] - g2[c].transform("max")
    f["n_s1_for_cand"] = g2["cheap"].transform("size")
    f["mutual_best"] = ((f["cheap_gap1"] == 0) & (f["cheap_gap2"] == 0)).astype(int)
    return f
