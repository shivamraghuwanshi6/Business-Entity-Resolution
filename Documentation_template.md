# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** MyTeam  
**Team Members:** [List all team members]  
**Submission Date:** 26 September 2026

---

## 1. Executive Summary
A supervised pairwise-classification pipeline over a small, scalable candidate
set. Records are normalised (transliteration, canonical abbreviations,
script-independent phonetic skeletons, legal-suffix removal), blocked with an
IDF-weighted sparse top-k search over discrete name/address keys, filtered by a
learned stage-1 model to **4.98 candidates per Source-1 entity
on test** (4.18 on train), and scored by a LightGBM matcher whose decision
threshold is tuned directly for macro F0.5. Out-of-fold macro F0.5 on train is
**0.9584** (India 0.9437, US 0.9682) against a blocking ceiling of 0.9743.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the full data (train: 2.21M S1 / 5.03M S2 / 5.29M S3 records; test:
1.73M / 4.89M / 5.08M):

- **Countries:** train has US (60 %) and India (40 %) in every source; test adds
  France (15 % of test S1). Labels are consistent across sources (`US`, `India`,
  `France`), none are missing, and every true train pair has the same country on
  both sides — so the country is a safe blocking bucket, handled as an open set.
- **Match structure:** 5.6 % of S1 entities are singletons; on average an S1
  entity has 3.46 matches (1.7 in S2, 1.8 in S3), i.e. S2/S3 contain several
  duplicates of the same business. 73–75 % of S2/S3 records match some S1.
- **Scripts:** S1 names are Latin script, but **15 % of S2 and 11 % of S3 names
  are written in Indic scripts** (Devanagari, Gujarati, Tamil, Kannada,
  Malayalam, Odia, Bengali). Transliterated with Unidecode they become e.g.
  `raam maarketting praaivett limittedd` ("Ram Marketing Private Limited"), so
  legal words were not recognised and the core names never aligned.
- **Name noise:** legal-suffix insertion/removal/reordering (`Pvt Ltd`,
  `LLC`, `(Corp.)`, `L.L.C.`, `SARL`, `EURL`), word transpositions, typos and
  leetspeak (`Hea1thcare`, `5ervices`, `c0rp`), web domains as names
  (`ridgefresenius.com`), store ids (`#36918`, `(ID: 78239)`), coined brand /
  DBA names that share nothing with the legal name (`Brixorbisol`, `Onyxecto`).
- **Address noise:** abbreviations (Rd/Road, BD/Boulevard, R./Rue), component
  reordering, missing components (3 % of S2/S3 addresses empty; `null`, `N/A`
  placeholders), state names in Indic script, landmark references
  (`Near SBI ATM`), and **house-number noise**: leading zeros (`0227`), letter
  suffixes (`849A`), a dropped digit (`3424`→`424`). Conversely, many *wrong*
  candidates are the same business name on the same street at a nearby number
  (`1197` vs `1201`) — house numbers turned out to be the single most important
  signal.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (learned candidate filter + LightGBM matcher)  
**Core Innovation:** (1) key-based IDF sparse top-k blocking that scales to
millions of records on a laptop CPU and reaches 0.94 pair recall at 18
candidates/entity; (2) script-independent normalisation (legal words of any
Indic script recognised through their consonant skeleton); (3) house-number
relation features that separate "same entity, noisy number" from "neighbouring
business on the same street"; (4) the decision rule is tuned on the exact
leaderboard metric (macro F0.5 including singletons) with grouped out-of-fold
predictions.

Pipeline:
1. **Normalisation** of names and addresses (parallel, cached).
2. **Blocking** — exact top-30 IDF-cosine neighbours over discrete keys in S2
   and in S3, within the same country bucket; exact char n-gram TF-IDF cosines;
   keep the top 10 per source by a blended cheap score.
3. **Stage-1 filter** — LightGBM on vectorised scores; its output *is*
   `candidate_pairs.tsv`.
4. **Stage-2 matcher** — LightGBM on ~75 name, address, number and context features.
5. **Decision rule** — per S1 entity keep candidates with probability ≥ t
   (and ≥ α × the entity's best probability); t = 0.70, α = 0 tuned for macro
   F0.5 on out-of-fold predictions.

Validation: `GroupKFold(5)` grouped by Source-1 id (no entity's candidates leak
across folds). Model fitting and OOF use a deterministic 30 % sample of train
S1 entities (661,897 entities, 3.0M candidate pairs) to fit in 16 GB RAM;
blocking and the stage-1 filter always run on all 2.2M.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** per record a bag of discrete keys — core-name tokens
  (legal forms removed), phonetic-skeleton tokens of the name (catches
  transliterations: Lakshmi/Laxmi, Hindi-script names), adjacent name-token
  bigrams, address words, house numbers, postal codes (5/6-digit) and address
  bigrams. Keys are TF-IDF weighted (sublinear TF, IDF over the country bucket,
  no labels) and keys shared by more than 0.05 % of the bucket (`pvt`, `road`,
  `delhi`) are dropped, so posting lists stay short. Retrieval is the **exact
  top-30 cosine neighbours** per S1 record in S2 and in S3 inside the same
  country bucket, computed with a multi-threaded sparse top-k product
  (`sparse_dot_topn`, Apache-2.0): cost is O(N·K·posting length), not O(N²);
  ~1 minute per million queries. The interface maps directly onto an ANN index
  at larger scale.
- **Scoring and pruning:** every retrieved pair gets exact character n-gram
  TF-IDF cosines of the core name (2–4-grams), the name skeleton (2–3-grams)
  and the address (3–4-grams); hashed features, IDF over the bucket. The list
  is pruned to the top 10 per source by `0.65·max(name, skeleton) + 0.35·address`.
- **Stage-1 learned filter:** LightGBM (15 leaves) on the cosines, the key
  score, the blended score, its rank and the gaps to the best candidate in both
  directions (best candidate for the S1 record, best S1 record for the
  candidate). The threshold keeps 98.5 % of the blocked true pairs; the
  surviving list is `candidate_pairs.tsv`.
- **Candidate pairs generated:**

| stage (train, all 2.2M S1) | pairs | avg candidates / S1 | pair recall |
|---|---|---|---|
| key retrieval (top-30 per source) | 132.4M | 60 | ≈0.95 (5k-query India benchmark) |
| after cheap-score pruning (10 per source) | 40.3M | 18.28 | 0.9399 |
| after stage-1 filter (= `candidate_pairs.tsv`) | 9.2M | **4.18** | **0.9260** |

  Test set: 18.44 per S1 after pruning, **4.98 per S1
  in `candidate_pairs.tsv`** (8,629,841 pairs for 1,732,544 S1).
  Reduction ratio vs. all S1×(S2∪S3) pairs (same-bucket or not): 0.99999823
  before the stage-1 filter and 0.99999959 after it on train; 0.99999950 for
  the submitted test `candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** the blocking ceiling is
  measured on every run (macro F0.5 if the matcher were perfect on the
  candidate set: **0.9743**). Keys were chosen from the error analysis of
  missed pairs: transliteration-robust skeleton keys, address keys that match
  even when the name is a coined brand, IDF weighting so rare keys dominate.
  Widening the pruned list from 6 to 10 per source and letting the *learned*
  filter do the final cut raised the ceiling from 0.9707 to 0.9774 (+0.005 OOF
  F0.5). The stage-1 recall target (98.5 %) was chosen as the smallest candidate
  set losing < 0.002 F0.5 (see §5 and Appendix B).

---

## 4. Matching Model

**Features used** (all country-agnostic — nothing is hard-coded per country,
so the model transfers to France):
- Name features: char n-gram TF-IDF cosines (core, skeleton); RapidFuzz ratio,
  partial ratio, token-sort and token-set ratios (core and full name);
  Jaro-Winkler; token Jaccard; IDF-weighted token overlap (min and Jaccard);
  skeleton-token Jaccard; space-insensitive ratio (web domains, glued words);
  first-token and exact-core equality; acronym match (IBM ↔ International
  Business Machines); containment; length difference; mean IDF of each name
  (flags coined brand names); non-Latin-script flag.
- Address features: TF-IDF cosine; fuzzy ratio / token-set / partial ratio; word
  Jaccard and IDF-weighted overlap; **house numbers**: number-set Jaccard,
  number conflict, first-number equality, Levenshtein similarity of the first
  numbers, one-digit-drop flag, numeric-core equality (`849` vs `849A`), log
  absolute difference; postal-code agreement (missing → NaN); city/tail
  equality; landmark flag; emptiness and length.
- Other: retrieval key score; context features — rank and gap to the best
  candidate of the same S1 record and source (for cosine, token-set, address,
  blended score), gap to the best S1 record for the same candidate (reverse
  view), mutual-best flag, number of competing candidates / S1 records, source
  flag, stage-1 probability.

Top features by gain: `a_num_jacc`, `a_num_conflict`, `p1` (stage-1
probability), `a_num_logdiff`, `a_num_drop`, `a_len_min`, `n_full_tset`,
`a_num_core_eq`, `n_idf_mean_r`, `n_s1_for_cand`.

**Normalisation details:** Unidecode transliteration (é→e, Indic scripts →
Latin); canonical short forms (Corporation→corp, Private→pvt, Limited→ltd,
Street→st, Road→rd, Boulevard/Bd/Bld→blvd, Rue/R.→r, Near→nr, Bengaluru→
bangalore …); transliterated legal words recognised via their consonant
skeleton (`praaivett`/`piraiveett`→pvt, `limittedd`/`limittett`→ltd,
`elelpii`→llp, `प्रा. लि.`→pvt ltd); legal forms removed for the core name
(Inc, LLC, Pvt Ltd, PC, SARL, SAS, EURL, EI, SELARL, SCP …); dotted acronyms
merged (L.L.C.→llc); leetspeak repaired inside words (hea1thcare→healthcare);
web domains and store ids stripped; French articles and `bis` dropped;
`null`/`N/A` placeholders removed; leading zeros stripped from house numbers;
phonetic skeleton (sh/ch→s, ksh/x→ks, w→v, doubled letters and inner vowels
removed).

**Model type:** LightGBM binary classifier (MIT licence, a few MB — far below
8B parameters): learning rate 0.1, 127 leaves, feature/bagging fraction 0.8,
early stopping per fold (≈1,700 rounds), final model retrained on all training
pairs with 1.1× the mean best iteration.  
**Threshold selection method:** grid search of (t, α) maximising macro F0.5 —
computed exactly like the leaderboard, singletons included — on the grouped
out-of-fold predictions; selected t = 0.70, α = 0.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9584** out-of-fold (GroupKFold(5) by S1 id,
  661,897 train S1 entities); India 0.9437, US 0.9682. Blocking ceiling 0.9743;
  predicting every entity empty would score 0.0557. On test the tuned rule
  predicts empty lists for 6.3 % of entities (train singleton
  share 5.6 %): India 6.5 %, US 5.9 %, France 6.6 %; 3.15 matches per entity on
  average (train truth: 3.46).

Experiments (each judged only by OOF macro F0.5; full log in
`artifacts/experiments.md`):

| # | change | blocking recall | cands / S1 | ceiling F0.5 | OOF F0.5 | India | US | kept |
|---|---|---|---|---|---|---|---|---|
| R0 | Baseline: original features, 6 per source, stage-1 recall 99.5 %, lr 0.05 / 63 leaves | 0.9150 | 4.34 | 0.9679 | 0.9383 | 0.9180 | 0.9519 | – |
| R1 | + normalisation v2 (transliterated legal words, leetspeak, acronyms, US/India/French forms, placeholders, leading zeros, web/ids) | 0.9218 | 4.38 | 0.9707 | 0.9440 | 0.9276 | 0.9550 | yes |
| R2 | + feature set v2 (house-number relations, name rarity, skeleton Jaccard, no-space ratio, script flag) | 0.9218 | 4.38 | 0.9707 | 0.9542 | 0.9363 | 0.9662 | yes |
| R3 | + digit-drop variants of house-number blocking keys | 0.9212 | 4.37 | 0.9704 | 0.9538 | 0.9361 | 0.9657 | no |
| R4 | + 10 per source, learned stage-1 pruning | 0.9399 | 4.79 | 0.9774 | 0.9591 | 0.9449 | 0.9686 | yes |
| R5 | + LightGBM lr 0.1 / 127 leaves | 0.9399 | 4.79 | 0.9774 | 0.9594 | 0.9455 | 0.9687 | yes |
| R6 | + retrieval score `cos_key` as a feature | 0.9399 | 4.63 | 0.9773 | 0.9596 | 0.9455 | 0.9690 | yes |
| R7 | + stage-1 recall 98.5 % (smaller candidate set) | 0.9399 | **4.18** | 0.9743 | **0.9584** | 0.9437 | 0.9682 | **yes — final** (−0.0012 < 0.002 budget) |
| R8 | stage-1 recall 98 % | 0.9399 | 4.08 | 0.9728 | 0.9571 | 0.9422 | 0.9671 | no (−0.0025 vs R6) |

Blocking recall = true-pair recall of the pruned list before the stage-1
filter, on all 2.2M train S1; cands / S1 = size of the final candidate list
(`candidate_pairs.tsv`). R7's settings are the pipeline defaults.

- **Common false positives (wrong merges):** (1) the same or a very similar name
  on the same street at a *nearby* house number (`Summit Center … 1197 Robeson
  St` vs `… 1201 Robeson St`), which looks like a branch of the same business;
  (2) same address with an unrelated or coined name (`Anchor` vs
  `Vantagedovaaria`); (3) a transliterated Indic name of a *different* business
  at the same address (`Royal Media` vs `रॉयल कंसल्टेंसी` = Royal Consultancy);
  (4) same name with an empty candidate address.
- **Common false negatives (missed matches):** (1) true matches whose house
  number was corrupted beyond a one-digit edit (`859` vs `962`) or whose street
  part is missing; (2) exact-name matches whose candidate address is empty
  (ambiguous by design — the model stays conservative because F0.5 rewards
  precision); (3) coined brand/DBA names at the same address (`Wheeler College`
  vs `ARCDELTANEX`); (4) candidate-set misses (6.0 % of true pairs lost in blocking, 7.4 % after the
  stage-1 filter), mostly coined names at the same address plus a dropped
  leading digit (`9224` vs `224`).

---

## 6. Conclusion
A scalable blocking + two-stage LightGBM pipeline reaches 0.958 out-of-fold
macro F0.5 with 4.2 candidates per entity on 2.2M training entities. The
largest gains came from looking at the data: script-independent normalisation of
legal words (+0.006) and house-number / name-rarity features (+0.010) mattered more
than model capacity, and letting a learned filter — rather than a hand-made
score — prune the candidate list recovered recall cheaply. Every step is
country-agnostic, which lets the same code handle France without training data.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (all source in `src/`, `README.md`,
`requirements.txt`, `run.sh`/`run.bat`). Entry point, run from
`student_resource/`:

```bash
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out-dir output --artifacts-dir artifacts
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (both
tab-separated) and `artifacts/run_report.json`. Modules: `prep.py` (loading,
parallel normalisation, parquet cache), `normalize.py` (rules), `blocking.py`
(keys, sparse top-k retrieval, exact cosines, pruning, stage-1 features),
`features.py` (vectorised pair and context features), `pipeline.py`
(orchestration, stage-1/stage-2 models, threshold tuning, outputs),
`error_analysis.py`, `tune_candidates.py`, `make_submission.py`. Runtime
≈ 1 h 45 min cold on a 20-core laptop CPU with 16 GB RAM (no GPU); every
run is deterministic.

### B. Additional Results
- Full experiment log with every change and its OOF score:
  `artifacts/experiments.md` (rejected: digit-drop blocking-key variants,
  −0.0004; stage-1 recall 98 %, −0.0025 vs best).
- Candidate-set trade-off (pre-screen on out-of-fold predictions, keep per source
  × stage-1 recall): F0.5 falls from 0.9588 (4.42 cands/S1) to 0.9539 at 6 per
  source and 0.9028 at 2 per source; details in
  `artifacts/runs/R4_candidate_tuning.tsv`.
- Per-country OOF F0.5: India 0.9437, US 0.9682. India is harder mainly because
  of the Indic-script names and landmark-style addresses.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
