#!/usr/bin/env bash
# Run from anywhere; assumes this folder sits at student_resource/code/business_entity_resolution
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
python3 "$HERE/src/pipeline.py" --data-dir dataset --out-dir output --artifacts-dir artifacts "$@"
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
