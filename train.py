"""Train SVM and Random Forest on the 63 landmark features.

Usage:
  python train.py                       # train.csv and test.csv from extract_features.py
  python train.py --no-curve            # skip the learning curve (faster)

The training images are split in blocks of frames (see results.py), 80% for training
and 20% for validation. The model that is better on validation is saved to
model.joblib for the webcam demo. The external test set is not used to choose anything:
the predictions on it are only saved to predictions/svm.csv and predictions/rf.csv.

The learning curve trains both models on growing parts of the training data and shows
the accuracy on the training part, on validation and on the external test.
"""
import argparse
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from results import FIG_DIR, is_val, save_predictions, test_images

FEATURES = [f"f{i}" for i in range(63)]
CURVE_PER_CLASS = [50, 100, 250, 500, 1000, 1500, 2000, "all"]


def make_models(probability=True):
    return {
        "svm": make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=10, gamma="scale",
                                 probability=probability, random_state=42)),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }


NAMES = {"svm": "SVM (RBF)", "rf": "Random Forest"}


def subset(df, per_class, seed=42):
    if per_class == "all":
        return df
    return pd.concat([g.sample(min(per_class, len(g)), random_state=seed)
                      for _, g in df.groupby("label")])


def learning_curve(train, val, test, test_y):
    rows = []
    for per_class in CURVE_PER_CLASS:
        part = subset(train, per_class)
        for key, model in make_models(probability=False).items():
            start = time.time()
            model.fit(part[FEATURES].values, part["label"].values)
            row = {"model": NAMES[key], "per_class": per_class, "n_train": len(part),
                   "train_acc": accuracy_score(part["label"], model.predict(part[FEATURES].values)),
                   "val_acc": accuracy_score(val["label"], model.predict(val[FEATURES].values)),
                   "test_acc_hand_found": accuracy_score(test_y, model.predict(test[FEATURES].values)),
                   "seconds": round(time.time() - start, 1)}
            rows.append(row)
            print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in row.items()), flush=True)
    curve = pd.DataFrame(rows)
    FIG_DIR.mkdir(exist_ok=True)
    curve.to_csv(FIG_DIR / "learning_curve_landmarks.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, g) in zip(axes, curve.groupby("model", sort=False)):
        ax.plot(g["n_train"], g["train_acc"], "o-", label="training part")
        ax.plot(g["n_train"], g["val_acc"], "o-", label="validation (same dataset)")
        ax.plot(g["n_train"], g["test_acc_hand_found"], "o-", label="external test (hand found)")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("training images with a hand (log scale)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right")
    fig.suptitle("Learning curves of the landmark models")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curve_landmarks.png", dpi=150)
    print(f"Saved {FIG_DIR / 'learning_curve_landmarks.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--test-dir", default="data/asl-alphabet-test")
    parser.add_argument("--no-curve", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    val_mask = df["file"].map(is_val)
    train, val = df[~val_mask], df[val_mask]
    test = pd.read_csv(args.test)
    print(f"Train: {len(train)}  Validation: {len(val)}  External test with a hand: {len(test)}")

    if not args.no_curve:
        print("\nLearning curve")
        learning_curve(train, val, test, test["label"].values)

    print("\nFinal models (all training images)")
    by_file = {}
    val_f1 = {}
    for key, model in make_models().items():
        start = time.time()
        model.fit(train[FEATURES].values, train["label"].values)
        pred = model.predict(val[FEATURES].values)
        val_f1[key] = f1_score(val["label"], pred, average="macro")
        print(f"{NAMES[key]:14s} validation accuracy {accuracy_score(val['label'], pred):.3f}  "
              f"macro F1 {val_f1[key]:.3f}  ({time.time() - start:.0f} s)")
        joblib.dump(model, f"{key}.joblib")
        by_file[key] = dict(zip(test["file"], model.predict(test[FEATURES].values)))

    print("\nSVM per-class report (validation):")
    print(classification_report(val["label"], joblib.load("svm.joblib").predict(val[FEATURES].values)))

    items = test_images(args.test_dir)
    files = [f.as_posix() for f, _ in items]
    for key in by_file:
        save_predictions(key, files, [l for _, l in items], [by_file[key].get(f) for f in files])

    best = max(val_f1, key=val_f1.get)
    joblib.dump(joblib.load(f"{best}.joblib"), "model.joblib")
    print(f"\n{NAMES[best]} is better on validation, saved as model.joblib for the demo")


if __name__ == "__main__":
    main()
