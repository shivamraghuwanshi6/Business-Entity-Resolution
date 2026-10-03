"""Pre-screen --keep-per-source / --stage1-recall settings from saved OOF pairs.

Tightening either setting only *removes* candidates, so its effect can be
approximated by masking the out-of-fold predictions of the last run (the
decision threshold is re-tuned for every setting). Context features would
shift slightly in a real re-run, so the chosen setting is confirmed with a
full pipeline run afterwards.

    python code/business_entity_resolution/src/tune_candidates.py --artifacts-dir artifacts
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import decide_mask, macro_f05  # noqa: E402
from prep import read_tsv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    a = ap.parse_args()
    oo = pd.read_parquet(os.path.join(a.artifacts_dir, "oof_pairs.parquet"))
    rep = json.load(open(os.path.join(a.artifacts_dir, "run_report.json")))
    gt = read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")).fillna("")
    # evaluation universe = sampled S1 (same deterministic sample as the run)
    from pipeline import in_sample
    s1 = gt["source1_entity_id"].values
    samp = in_sample(s1, rep["args"]["train_frac"])
    ids = s1[samp]
    ntrue = gt["matched_entity_ids"].values[samp]
    ntrue = np.array([len([x for x in v.split(",") if x]) for v in ntrue], float)
    code = pd.Series(np.arange(len(ids)), index=ids).reindex(oo["s1_id"].values).values
    y, p, p1, rk = oo["y"].values, oo["p"].values, oo["p1"].values, oo["cheap_rank"].values
    pos = np.sort(p1[y == 1])
    base_kps = int(rk.max()) + 1
    print(f"S1 evaluated={len(ids)}  pairs={len(oo)}  kps in run={base_kps}")
    print("kps  s1_recall  cands/S1  F0.5    (thr, alpha)")
    rows = []
    for kps in range(2, base_kps + 1):
        for rec in (0.995, 0.99, 0.98, 0.97, 0.95):
            t1 = pos[int((1 - rec) * len(pos))]
            m = (rk < kps) & (p1 >= t1)
            best = (-1, 0, 0)
            for t in np.arange(0.3, 0.96, 0.025):
                for al in (0.0, 0.3, 0.5):
                    s = macro_f05(code[m], y[m], decide_mask(code[m], p[m], t, al), ntrue)
                    if s > best[0]:
                        best = (s, t, al)
            rows.append((kps, rec, m.sum() / len(ids), *best))
            print(f"{kps:>3}  {rec:>9}  {m.sum() / len(ids):8.2f}  {best[0]:.4f}  ({best[1]:.3f}, {best[2]})",
                  flush=True)
    pd.DataFrame(rows, columns=["kps", "stage1_recall", "cands", "f05", "t", "alpha"]).to_csv(
        os.path.join(a.artifacts_dir, "candidate_tuning.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
