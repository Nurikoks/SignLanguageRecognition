"""Take the clips of our words (words.py) out of the Kaggle ISLR archive and keep only
the hands and the pose.

Usage:
  kaggle competitions download -c asl-signs -p data/islr/full     # about 40 GB, once
  python islr_prepare.py
  python islr_prepare.py --zip data/islr/full/asl-signs.zip --workers 8

The competition is "asl-signs" (Google Isolated Sign Language Recognition, 250 ASL
signs, 21 people). Its rules must be accepted on kaggle.com first. Single-file downloads
are rate-limited by Kaggle, so we download the whole archive and read only the files
we need from it (no need to unpack all of it). The archive can be deleted afterwards.

Each clip is a parquet file with 543 MediaPipe Holistic landmarks per frame (face 468,
left hand 21, pose 33, right hand 21). We save frames x 75 x 3 (left hand, right hand,
pose; NaN where a part was not found) as float16 to data/islr/clips/<sequence_id>.npy.
data/islr/clips.csv lists the clips of our words (path, participant_id, sequence_id, sign).
"""
import argparse
import io
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from words import ISLR_DIR, WORDS

CLIPS = ISLR_DIR / "clips"
PARTS = [("left_hand", 21), ("right_hand", 21), ("pose", 33)]
_zip = None


def to_array(df):
    frames = np.sort(df["frame"].unique())
    out = np.full((len(frames), 75, 3), np.nan, np.float32)
    row = {f: i for i, f in enumerate(frames)}
    start = 0
    for part, n in PARTS:
        d = df[df["type"] == part]
        out[d["frame"].map(row).to_numpy(), start + d["landmark_index"].to_numpy()] = \
            d[["x", "y", "z"]].to_numpy()
        start += n
    return out


def _convert(job):
    global _zip
    zip_path, member, target = job
    if Path(target).exists():
        return True
    if _zip is None:                  # one open archive per worker process
        _zip = zipfile.ZipFile(zip_path)
    df = pd.read_parquet(io.BytesIO(_zip.read(member)), columns=["frame", "type",
                                                               "landmark_index", "x", "y", "z"])
    np.save(target, to_array(df[df["type"] != "face"]).astype(np.float16))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(ISLR_DIR / "full" / "asl-signs.zip"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as z:
        names = set(z.namelist())
        if not (ISLR_DIR / "train.csv").exists():
            (ISLR_DIR / "train.csv").write_bytes(z.read("train.csv"))
    meta = pd.read_csv(ISLR_DIR / "train.csv")
    meta = meta[meta["sign"].isin(WORDS)].reset_index(drop=True)
    missing = set(WORDS) - set(meta["sign"])
    if missing:
        raise SystemExit(f"Not in the dataset: {sorted(missing)}")
    absent = [p for p in meta["path"] if p not in names]
    if absent:
        raise SystemExit(f"{len(absent)} clips are not in the archive, e.g. {absent[0]}")

    CLIPS.mkdir(parents=True, exist_ok=True)
    jobs = [(args.zip, p, str(CLIPS / f"{sid}.npy"))
            for p, sid in zip(meta["path"], meta["sequence_id"])]
    print(f"{len(jobs)} clips of {len(WORDS)} words from {meta['participant_id'].nunique()} people")
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_convert, jobs, chunksize=32), 1):
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    meta.to_csv(ISLR_DIR / "clips.csv", index=False)
    print(f"Saved {len(meta)} clips to {CLIPS} and the list to {ISLR_DIR / 'clips.csv'}")
    print(meta["sign"].value_counts().describe()[["min", "mean", "max"]].round(0).to_dict())


if __name__ == "__main__":
    main()
