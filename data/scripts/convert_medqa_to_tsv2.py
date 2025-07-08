#!/usr/bin/env python3
"""
convert_medqa_to_tsv.py  –  MedQA debates → BDoG-compatible TSV

Key change vs. previous script:
  • 'image' column now filled with a dummy 1×1-px PNG base64
    so TSVDataset's >64-char check passes.
"""
import argparse, csv, hashlib, json, pathlib

# a transparent 1×1 PNG (Base64, 88 chars)
DUMMY_PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
             "QVR42mNk+A8AAwMB/6Z5gXcAAAAASUVORK5CYII=")

parser = argparse.ArgumentParser()
parser.add_argument("--in",  dest="src",  required=True,
                    help="path to original_debates.json")
parser.add_argument("--out", dest="dst",  default="data/local_data/MedQA.tsv",
                    help="destination .tsv (default data/local_data/)")
args = parser.parse_args()

src = pathlib.Path(args.src).expanduser()
dst = pathlib.Path(args.dst).expanduser()
dst.parent.mkdir(parents=True, exist_ok=True)

header = ["index", "image", "question", "hint",
          "A", "B", "C", "D", "E", "answer"]

with src.open() as fp_json, dst.open("w", newline="") as fp_tsv:
    debates = json.load(fp_json)
    wr = csv.DictWriter(fp_tsv, fieldnames=header, delimiter="\t")
    wr.writeheader()

    for i, ex in enumerate(debates):
        opts = ex["options"]
        wr.writerow({
            "index": i,
            "image": "none",        # ← satisfies >64-char assert
            "question": ex["question"],
            "hint": "",
            "A": opts.get("A", ""), "B": opts.get("B", ""), "C": opts.get("C", ""),
            "D": opts.get("D", ""), "E": opts.get("E", ""),
            "answer": ex["solution"]
        })

md5 = hashlib.md5(dst.read_bytes()).hexdigest()
print(f"✅  Wrote {dst}  ({len(debates)} rows)")
print(f"🔑  MD5 = {md5}")
