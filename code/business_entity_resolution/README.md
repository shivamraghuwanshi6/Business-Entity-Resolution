# Business Entity Resolution — Amazon ML Challenge 2026

Key-based sparse blocking + learned stage-1 filter + gradient-boosted pairwise
matcher. Regenerates both `output/candidate_pairs.tsv` and
`output/matching_results.tsv` from the raw train/test TSVs. Only the provided
data is used — no external data, APIs, geocoding or pretrained models.
The only model is LightGBM (MIT licence, a few MB; far below 8B parameters).

## Layout expected

```
student_resource/
├── dataset/train/  train_source{1,2,3}.tsv, train_ground_truth.tsv
├── dataset/test/   test_source{1,2,3}.tsv
├── utils/validate_submission.py
├── Documentation_template.md
└── code/business_entity_resolution/   <- this folder
    ├── src/  pipeline.py      end-to-end entry point
    │         prep.py          loading + parallel normalisation (parquet cache)
    │         normalize.py     name/address normalisation rules
    │         blocking.py      key-based sparse top-k retrieval, exact cosines, pruning
    │         features.py      vectorised pair features + context features
    │         error_analysis.py   OOF false positives / negatives / blocking misses
    │         tune_candidates.py  pre-screen --keep-per-source / --stage1-recall
    │         make_submission.py  builds <team>_submission.zip
    ├── requirements.txt  run.sh  run.bat
```

## Reproduce end-to-end

```bash
cd student_resource
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py \
    --data-dir dataset --out-dir output --artifacts-dir artifacts
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
(or just `code/business_entity_resolution/run.sh` / `run.bat`). The defaults
are the final configuration; no flags are needed.

One run does:
1. **normalise** all six source files in parallel (cached as parquet under
   `artifacts/cache/`, keyed by a hash of `normalize.py`);
2. **block** every (country bucket × target source) chunk: sparse TF-IDF over
   discrete keys (name tokens, phonetic skeletons, name bigrams, address words,
   house numbers, postal codes, address bigrams), exact top-30 per source with a
   multi-threaded sparse top-k product, exact char n-gram cosines, keep the top
   10 per source; then compute the pair features (cached under `artifacts/work/`);
3. train the **stage-1 filter** (grouped OOF) and cut the candidate list at the
   threshold keeping 98.5 % of blocked true pairs → this list *is*
   `candidate_pairs.tsv`;
4. train the **stage-2 matcher** with 5-fold GroupKFold (grouped by Source-1 id)
   and tune the decision threshold for macro F0.5 on the out-of-fold predictions;
5. retrain on all training pairs, then block → filter → score the test set and
   write both TSVs.

Diagnostics go to `artifacts/run_report.json` (blocking recall, candidates per
entity, OOF F0.5 overall and per country, blocking ceiling, feature importance),
the out-of-fold pairs to `artifacts/oof_pairs.parquet`, the models to
`artifacts/model_stage{1,2}.txt`.

Model fitting / OOF evaluation use a deterministic 30 % sample of the training
Source-1 entities (`--train-frac 0.3`, ~662k entities, ~3M pairs) to fit in
16 GB RAM; blocking, recall and the stage-1 filter always cover all 2.2M.

**Runtime** (20-core laptop CPU, 16 GB RAM, no GPU needed): ~1 h 45 min for a
cold run (normalisation ~10 min, train blocking+features ~30 min, models
~20 min, test blocking+features+scoring ~35 min). Re-runs reuse the caches:
changing only model/candidate settings takes ~25 min.

### Useful flags

| flag | default | meaning |
|---|---|---|
| `--keep-per-source` | 10 | candidates kept per S1 record and source before the stage-1 filter |
| `--stage1-recall` | 0.985 | share of blocked true pairs the stage-1 filter must keep |
| `--k-key` | 30 | neighbours retrieved per S1 record and source |
| `--key-max-df` | 0.0005 | drop blocking keys shared by more than this share of a bucket |
| `--feature-set` | v2 | `v1` = original feature set |
| `--key-feature/--no-key-feature` | on | retrieval score as a feature |
| `--lr`, `--num-leaves`, `--max-rounds` | 0.1, 127, 3000 | stage-2 LightGBM |
| `--train-frac` | 0.3 | share of train S1 used for model fitting / OOF |
| `--prune-score`, `--num-variants` | cheap, off | alternatives tried and rejected (see `artifacts/experiments.md`) |
| `--cross-country` | off | ignore the country label in blocking |
| `--skip-test` | off | train + validate only |
| `--rebuild` | off | ignore cached blocking/feature chunks |

Deterministic: fixed seeds, deterministic S1 sample.

## Licences

The only trained model is LightGBM (MIT). Libraries: pandas/numpy/scipy/
scikit-learn (BSD), rapidfuzz (MIT), pyarrow and sparse_dot_topn (Apache-2.0),
Unidecode (GPL-2.0+, used only as a text transliteration library, not a model).

## Analysis helpers

```bash
python code/business_entity_resolution/src/error_analysis.py --artifacts-dir artifacts
python code/business_entity_resolution/src/tune_candidates.py --artifacts-dir artifacts
```

## Package the final submission

```bash
python code/business_entity_resolution/src/make_submission.py --team <team_name>
```
