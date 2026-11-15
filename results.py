"""Things shared by all letter models: the train/validation split, the external test
images and the prediction files that evaluate_all.py reads.

Every model script writes predictions/<model>.csv with one row per external test image
(720 rows): file, true letter, predicted letter. The prediction is empty when the model
gave no answer (MediaPipe found no hand). evaluate_all.py computes every number and
figure of the report from these files, so all models are compared on the same images.
"""
import re
from pathlib import Path

import pandas as pd

from features import LETTERS

TRAIN_DIR = "data/asl_alphabet_train"
TEST_DIR = "data/asl-alphabet-test"
PRED_DIR = Path("predictions")
FIG_DIR = Path("figures")
IMAGE_EXT = (".jpg", ".jpeg", ".png")

# The training images are frames of videos, so neighbouring frames are almost the same
# picture. Frames are split in blocks of 50 in a row, every 5th block goes to validation
# (20%). A random split would put near copies in both parts and the validation score
# would be too high.
BLOCK = 50
VAL_EVERY = 5


def is_val(path):
    number = int(re.search(r"\d+", Path(path).stem).group())
    return (number // BLOCK) % VAL_EVERY == 0


def list_images(folder):
    return sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in IMAGE_EXT)


def test_images(test_dir=TEST_DIR):
    """The 720 external test images in a fixed order: [(path, letter)]."""
    from extract_features import find_class_folders
    folders = find_class_folders(test_dir)
    if sorted(folders) != LETTERS:
        raise SystemExit(f"Need a folder for each of the 24 letters in {test_dir}")
    return [(f, letter) for letter in LETTERS for f in list_images(folders[letter])]


def save_predictions(name, files, true, pred):
    """pred: predicted letter or None (no answer)."""
    PRED_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame({"file": [Path(f).as_posix() for f in files], "true": true,
                       "pred": ["" if p is None else p for p in pred]})
    path = PRED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} test predictions to {path}")
