#!/usr/bin/env python3
"""
convert_medqa_to_tsv.py
───────────────────────
Convert the MedQA Google-MAD multi-agent debate file
`original_debates.json` into the TSV layout required by the BDoG repo.

Usage
─────
    python convert_medqa_to_tsv.py \
           --in  original_debates.json \
           --out MedQA.tsv
"""
import argparse, csv, hashlib, json, pathlib, sys

# ------------------------------------------------------------
# 1. command-line arguments
# ------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--in",  dest="infile",
                required=True,  help="path to original_debates.json")
ap.add_argument("--out", dest="outfile",
                default="MedQA.tsv", help="destination TSV file")
args = ap.parse_args()

src  = pathlib.Path(args.infile).expanduser()
dest = pathlib.Path(args.outfile).expanduser()
dest.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. TSV column names – exactly what BDoG's TSVDataset expects
# ------------------------------------------------------------
header = ["index", "image", "question", "hint",
          "A", "B", "C", "D", "E", "answer"]

# ------------------------------------------------------------
# 3. load JSON and write TSV
# ------------------------------------------------------------
with src.open() as fp_json, dest.open("w", newline="") as fp_tsv:
    data   = json.load(fp_json)          # full list of debates
    writer = csv.DictWriter(fp_tsv, fieldnames=header, delimiter="\t")
    writer.writeheader()

    for i, ex in enumerate(data):
        # --------- 3.1  guarantee every choice column exists ----------
        opts = ex["options"]                         # dict of letters → text
        row  = {letter: opts.get(letter, "")         # blank if missing
                for letter in ["A", "B", "C", "D", "E"]}

        # --------- 3.2  mandatory BDoG fields -------------------------
        row.update({
            "index":    i,                 # sequential id
            "image":    "none",            # text-only dataset
            "question": ex["question"],
            "hint":     "",                # no extra hint in MedQA
            "answer":   ex["solution"]     # gold letter
        })
        writer.writerow(row)

# ------------------------------------------------------------
# 4. print MD5 so you can register the dataset in BDoG
# ------------------------------------------------------------
md5 = hashlib.md5(dest.read_bytes()).hexdigest()
print(f"✅   Wrote {dest}  ({len(data)} lines)")
print(f"🔑   MD5 = {md5}")
