"""Print OOF false positives / false negatives and true pairs missed by blocking.

Run from student_resource/ after a pipeline run:
    python code/business_entity_resolution/src/error_analysis.py --artifacts-dir artifacts
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import read_tsv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--work", default=None, help="chunk dir (default: newest under work/)")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    oo = pd.read_parquet(os.path.join(a.artifacts_dir, "oof_pairs.parquet"))
    fp = oo[oo.pred & (oo.y == 0)]
    fn = oo[~oo.pred & (oo.y == 1)]
    fp_s = fp.sample(min(a.n, len(fp)), random_state=a.seed)
    fn_s = fn.sample(min(a.n, len(fn)), random_state=a.seed)

    # true pairs of the evaluated S1 that never reached the candidate set
    work = a.work or max(glob.glob(os.path.join(a.artifacts_dir, "work", "*")), key=os.path.getmtime)
    blocked = pd.concat([pd.read_parquet(f, columns=["s1_id", "entity_id"])
                         for f in glob.glob(os.path.join(work, "train_*.parquet"))])
    blocked = blocked[blocked.s1_id.isin(set(oo.s1_id))]
    key = set(zip(blocked.s1_id, blocked.entity_id))
    del blocked
    gt = read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")).fillna("")
    gt = gt[gt.source1_entity_id.isin(set(oo.s1_id))]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m != ""]
    miss = ex[[(s, m) not in key for s, m in zip(ex.source1_entity_id, ex.m)]]
    miss_s = miss.sample(min(a.n, len(miss)), random_state=a.seed)

    need = set(fp_s.s1_id) | set(fp_s.entity_id) | set(fn_s.s1_id) | set(fn_s.entity_id)         | set(miss_s.source1_entity_id) | set(miss_s.m)
    parts = []
    for i in (1, 2, 3):
        for ch in pd.read_csv(os.path.join(a.data_dir, "train", f"train_source{i}.tsv"), sep="\t",
                              dtype=str, keep_default_na=False, quoting=3, chunksize=500000):
            parts.append(ch[ch.entity_id.isin(need)])
    raw = pd.concat(parts).set_index("entity_id")
    name, addr = raw["business_name"], raw["business_address"]

    def show(df, title):
        print(f"\n===== {title} ({len(df)} shown)")
        for r in df.itertuples():
            print(f"[{r.bucket}] p={r.p:.3f} cos_n={r.cos_name:.2f} cos_a={r.cos_addr:.2f} "
                  f"tset={r.n_tset:.0f}/{r.a_tset if r.a_tset == r.a_tset else -1:.0f}")
            print(f"   S1: {name.get(r.s1_id)!s:45.45} | {addr.get(r.s1_id)!s:.80}")
            print(f"   {r.entity_id[:2]}: {name.get(r.entity_id)!s:45.45} | {addr.get(r.entity_id)!s:.80}")

    print(f"OOF pairs={len(oo)} FP={len(fp)} FN(in candidates)={len(fn)} TP={int((oo.pred & (oo.y == 1)).sum())}")
    print("FP by bucket:", fp.bucket.value_counts().to_dict(), " FN by bucket:", fn.bucket.value_counts().to_dict())
    show(fp_s, "FALSE POSITIVES")
    show(fn_s, "FALSE NEGATIVES (in candidate set)")
    print(f"\n===== MISSED BY BLOCKING: {len(miss)} of {len(ex)} true pairs of evaluated S1 "
          f"({len(miss) / max(1, len(ex)):.3%})")
    for r in miss_s.itertuples():
        print(f"   S1: {name.get(r.source1_entity_id)!s:45.45} | {addr.get(r.source1_entity_id)!s:.80}")
        print(f"   {r.m[:2]}: {name.get(r.m)!s:45.45} | {addr.get(r.m)!s:.80}")


if __name__ == "__main__":
    main()
