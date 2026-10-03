@echo off
REM Run from anywhere; assumes this folder sits at student_resource\code\business_entity_resolution
set HERE=%~dp0
cd /d "%HERE%..\.."
python "%HERE%src\pipeline.py" --data-dir dataset --out-dir output --artifacts-dir artifacts %*
if errorlevel 1 exit /b 1
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
