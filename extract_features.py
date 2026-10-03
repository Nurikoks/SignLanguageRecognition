"""Run MediaPipe on every image in a dataset folder and save 63 features per image to CSV.

Usage:
  python extract_features.py --data data/asl_alphabet_train --out train.csv --per-class 500
  python extract_features.py --data data/asl-alphabet-test --out test.csv
"""
import argparse
import random
from pathlib import Path

import cv2
import pandas as pd

from features import LETTERS, create_detector, detect, normalize


def find_class_folders(root):
    """Find folders named A, B, C ... (SPACE, DELETE, NOTHING, J and Z are skipped)."""
    folders = {}
    for d in sorted(Path(root).rglob("*")):
        if d.is_dir() and len(d.name) == 1 and d.name.upper() in LETTERS:
            folders.setdefault(d.name.upper(), d)
    return folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=None,
                        help="use only N random images per letter (faster)")
    args = parser.parse_args()

    folders = find_class_folders(args.data)
    print(f"Found {len(folders)} letter folders")
    detector = create_detector()
    rows, total, skipped = [], 0, 0

    for letter, folder in folders.items():
        files = [f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if args.per_class and len(files) > args.per_class:
            random.seed(42)
            files = random.sample(files, args.per_class)
        found = 0
        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            total += 1
            points, side = detect(detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if points is None:
                skipped += 1
                continue
            rows.append(list(normalize(points, side)) + [letter])
            found += 1
        print(f"{letter}: {found}/{len(files)} images with a hand")

    df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(63)] + ["label"])
    df.to_csv(args.out, index=False)
    print(f"\nImages processed: {total}, no hand found: {skipped}")
    print(f"Saved {len(df)} samples x 63 features to {args.out}")
    if len(df):
        first = df.iloc[0]
        print(f"Example ({first['label']}), first 6 of 63 values:",
              [round(v, 3) for v in first.iloc[:6]])


if __name__ == "__main__":
    main()
