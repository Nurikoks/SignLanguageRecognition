"""Train the word model on recordings from record_words.py (data/words/<WORD>/*.npz).

Usage:
  python train_words.py

There are only a few examples per word, so the models are compared with
cross-validation, and the training examples are augmented: the live demo looks at
a sliding window, so the gesture may start a bit earlier or later in it.
"""
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gestures import WORDS_DIR, WORDS_MODEL, sequence_features

COPIES = 5        # augmented copies per training example
MAX_CUT = 0.15    # cut up to 15% of the frames from the start and from the end


def load_recordings(root):
    recordings, labels = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(folder.glob("*.npz")):
            data = np.load(f)
            recordings.append((data["points"], data["present"]))
            labels.append(folder.name)
    return recordings, np.array(labels)


def features(recordings, labels, copies=0, seed=42):
    """Features of each recording, plus `copies` randomly cut and shifted versions."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for (points, present), label in zip(recordings, labels):
        versions = [(points, present)]
        for _ in range(copies):
            n = len(points)
            a = rng.integers(0, int(n * MAX_CUT) + 1)
            b = n - rng.integers(0, int(n * MAX_CUT) + 1)
            shift = rng.normal(0, 15, size=2).astype(np.float32)   # move hands a bit (pixels)
            cut = points[a:b].copy()
            cut[..., :2] += shift
            versions.append((cut, present[a:b]))
        for p, s in versions:
            x = sequence_features(p, s)
            if x is not None:
                X.append(x)
                y.append(label)
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(WORDS_DIR))
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"{root} not found. Record words first: python record_words.py HELLO")
    recordings, labels = load_recordings(root)
    counts = Counter(labels.tolist())
    print("Examples per word:", dict(counts))
    if len(counts) < 2:
        raise SystemExit("Need at least 2 words (or 1 word + 'nothing') to train")
    if min(counts.values()) < 3:
        raise SystemExit("Need at least 3 examples of every word (better 15-20)")

    models = {
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", C=10, gamma="scale",
                                       probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=min(5, min(counts.values())), shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        pred = np.empty(len(labels), dtype=object)
        for train_idx, test_idx in cv.split(np.zeros(len(labels)), labels):
            X_train, y_train = features([recordings[i] for i in train_idx], labels[train_idx],
                                        copies=COPIES)
            m = clone(model).fit(X_train, y_train)
            for i in test_idx:
                x = sequence_features(*recordings[i])
                pred[i] = m.predict([x])[0] if x is not None else None
        ok = pred != None  # noqa: E711 (element-wise check)
        results[name] = (accuracy_score(labels[ok], pred[ok].astype(str)),
                         f1_score(labels[ok], pred[ok].astype(str), average="macro"), pred)
        print(f"{name:14s} cross-validation accuracy {results[name][0]:.3f}  "
              f"macro F1 {results[name][1]:.3f}")

    best = max(results, key=lambda n: results[n][1])
    pred = results[best][2]
    ok = pred != None  # noqa: E711
    print(f"\n{best} per-word report (cross-validation):")
    print(classification_report(labels[ok], pred[ok].astype(str), zero_division=0))

    X, y = features(recordings, labels, copies=COPIES)
    joblib.dump(clone(models[best]).fit(X, y), WORDS_MODEL)
    print(f"Saved {best} trained on all {len(recordings)} examples to {WORDS_MODEL}")


if __name__ == "__main__":
    main()
