"""Run MediaPipe on every image in a dataset folder and save 63 features per image to CSV.

Usage:
  python extract_features.py --data data/asl_alphabet_train --out train.csv
  python extract_features.py --data data/asl-alphabet-test --out test.csv
  python extract_features.py --data data/asl_alphabet_train --out train.csv --per-class 500

Each row also keeps the image path (column "file"). train.py uses it to split the
training frames in blocks (like train_mobilenet.py), and evaluate_all.py uses it to
match the predictions of all models on the same test images.
Images where no hand is found are not saved.
"""
import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from features import LETTERS

_detector = None


def find_class_folders(root):
    """Find folders named A, B, C ... (SPACE, DELETE, NOTHING, J and Z are skipped).
    If a letter has two folders (the test download has a copy of itself inside),
    the first one is used."""
    folders = {}
    for d in sorted(Path(root).rglob("*")):
        if d.is_dir() and len(d.name) == 1 and d.name.upper() in LETTERS:
            folders.setdefault(d.name.upper(), d)
    return folders


def _features(path):
    """Worker: 63 features of one image, or None if there is no hand."""
    global _detector
    from features import create_detector, detect, normalize
    if _detector is None:          # one MediaPipe detector per worker process
        _detector = create_detector()
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    points, side = detect(_detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return None if points is None else list(normalize(points, side))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=None,
                        help="use only N random images per letter (faster)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    folders = find_class_folders(args.data)
    print(f"Found {len(folders)} letter folders")
    jobs = []
    for letter, folder in folders.items():
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if args.per_class and len(files) > args.per_class:
            files = sorted(random.Random(42).sample(files, args.per_class))
        jobs += [(f, letter) for f in files]

    with Pool(args.workers) as pool:
        feats = pool.map(_features, [f for f, _ in jobs], chunksize=64)

    rows, per_letter = [], {}
    for (f, letter), x in zip(jobs, feats):
        found, total = per_letter.get(letter, (0, 0))
        per_letter[letter] = (found + (x is not None), total + 1)
        if x is not None:
            rows.append(x + [letter, f.as_posix()])
    for letter, (found, total) in per_letter.items():
        print(f"{letter}: {found}/{total} images with a hand")

    df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(63)] + ["label", "file"])
    df.to_csv(args.out, index=False)
    print(f"\nImages processed: {len(jobs)}, no hand found: {len(jobs) - len(df)}")
    print(f"Saved {len(df)} samples x 63 features to {args.out}")


if __name__ == "__main__":
    main()
