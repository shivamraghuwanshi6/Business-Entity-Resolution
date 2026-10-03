"""Loading + normalisation of the source files, parallel and cached.

Normalising ~12M records is the most CPU-heavy pure-Python step, so it runs on
all cores and the result is cached as parquet keyed by a hash of
normalize.py (changing a rule automatically invalidates the cache).
"""
import hashlib
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import normalize

CACHE_VERSION = "3"  # bump when the cached schema changes

HERE = os.path.dirname(os.path.abspath(__file__))

NAME_KEYS = ("core", "full", "skel", "acro", "first")
ADDR_KEYS = ("full", "words", "nums", "first_num", "postal", "landmark", "tail")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
                       quoting=3)


def country_key(c) -> str:
    return c.strip().lower() if isinstance(c, str) and c.strip() else "__missing__"


def _norm_chunk(args):
    names, addrs = args
    out = {f"n_{k}": [] for k in NAME_KEYS}
    out.update({f"a_{k}": [] for k in ADDR_KEYS})
    out["n_script"] = []
    for n, a in zip(names, addrs):
        # name written in a non-Latin script (Devanagari, Tamil, ...): transliterated
        out["n_script"].append(int(any(ord(ch) > 0x24F for ch in n)))
        dn = normalize.norm_name(n)
        da = normalize.norm_addr(a)
        for k in NAME_KEYS:
            out[f"n_{k}"].append(dn[k])
        for k in ADDR_KEYS:
            v = da[k]
            if isinstance(v, (set, list)):
                v = " ".join(sorted(v) if isinstance(v, set) else v)
            out[f"a_{k}"].append(v)
    return out


def _rules_hash():
    with open(os.path.join(HERE, "normalize.py"), "rb") as fh:
        return hashlib.md5(fh.read() + CACHE_VERSION.encode()).hexdigest()[:10]


def ensure_cache(path, cache_dir, pool=None, chunk=20000):
    """Normalise one source TSV into a parquet cache; return the cache path."""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    cpath = os.path.join(cache_dir, f"{base}_{_rules_hash()}.parquet")
    if os.path.exists(cpath) and os.path.getmtime(cpath) >= os.path.getmtime(path):
        return cpath
    raw = read_tsv(path)
    n = len(raw)
    names = raw["business_name"].fillna("").tolist()
    addrs = raw["business_address"].fillna("").tolist()
    ids = raw["entity_id"].values
    ckey = (raw["country"].map(country_key).values if "country" in raw
            else np.full(n, "__missing__", dtype=object))
    del raw
    jobs = ((names[s:s + chunk], addrs[s:s + chunk]) for s in range(0, n, chunk))
    own = pool is None
    pool = pool or Pool(max(1, (os.cpu_count() or 2) - 2))
    tmp = cpath + ".tmp"
    writer, s = None, 0
    # stream each normalised chunk straight to parquet (bounded memory)
    for part in pool.imap(_norm_chunk, jobs):
        m = len(part["n_core"])
        df = pd.DataFrame({"row": np.arange(s, s + m, dtype=np.int64), "entity_id": ids[s:s + m]})
        for k, v in part.items():
            df[k] = np.asarray(v, dtype=np.int8) if k in ("a_landmark", "n_script") else v
        df["ckey"] = ckey[s:s + m]
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        writer = writer or pq.ParquetWriter(tmp, tbl.schema)
        writer.write_table(tbl)
        s += m
    if writer:
        writer.close()
    if own:
        pool.close()
    os.replace(tmp, cpath)
    return cpath


def bucket_keys(cpath):
    return sorted(set(pq.read_table(cpath, columns=["ckey"]).column("ckey").to_pylist()))


def read_bucket(cpath, bucket, with_missing=False):
    """Arrow table of one country bucket (optionally + records without a label).

    The "__missing__" bucket (records without a country) is compared with all.
    """
    t = pq.read_table(cpath)
    if bucket != "__missing__":
        keys = [bucket, "__missing__"] if with_missing else [bucket]
        t = t.filter(pc.is_in(t.column("ckey"), pa.array(keys)))
    return t.combine_chunks()


def col(t, name, idx=None):
    """Column of an arrow table as a numpy array (object for strings)."""
    c = t.column(name)
    if idx is not None:
        c = c.take(pa.array(idx))
    return c.to_numpy(zero_copy_only=False)


def iter_col(t, name, batch=500000):
    c = t.column(name)
    for s in range(0, len(c), batch):
        yield from c.slice(s, batch).to_pylist()
