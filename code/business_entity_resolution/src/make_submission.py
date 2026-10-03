"""Build <team>_submission.zip in the structure the organisers require.

Run from the student_resource/ folder:
    python code/business_entity_resolution/src/make_submission.py --team MyTeam
"""
import argparse
import os
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--team", required=True)
ap.add_argument("--root", default=".")
a = ap.parse_args()
root = os.path.abspath(a.root)
need = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md"]
for n in need:
    if not os.path.exists(os.path.join(root, n)):
        raise SystemExit(f"missing {n} - run the pipeline / fill the template first")
out = os.path.join(root, f"{a.team}_submission.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for n in need:
        z.write(os.path.join(root, n), n)
    code = os.path.join(root, "code", "business_entity_resolution")
    for d, _, files in os.walk(code):
        if "__pycache__" in d:
            continue
        for f in files:
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, root))
print("wrote", out)
