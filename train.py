"""Train SVM and Random Forest on landmark features and compare them.

Usage:
  python train.py --train train.csv --test test.csv
"""
import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load(path):
    df = pd.read_csv(path)
    return df.drop(columns="label").values, df["label"].values


def scores(model, X, y):
    pred = model.predict(X)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "macro_f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default=None, help="external test set (ASL Alphabet Test)")
    args = parser.parse_args()

    X, y = load(args.train)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(X_train)}  Test split: {len(X_val)}  Classes: {len(set(y))}")
    ext = load(args.test) if args.test else None

    models = {
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", C=10, gamma="scale",
                                       probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append({"model": name, "data": "test split", **scores(model, X_val, y_val)})
        if ext is not None:
            results.append({"model": name, "data": "external test", **scores(model, *ext)})

    table = pd.DataFrame(results).round(3)
    print("\n" + table.to_string(index=False))

    svm = models["SVM (RBF)"]
    print("\nSVM per-class report (test split):")
    print(classification_report(y_val, svm.predict(X_val), zero_division=0))

    ConfusionMatrixDisplay.from_estimator(svm, X_val, y_val, cmap="Blues", colorbar=False)
    plt.title("SVM confusion matrix (test split)")
    plt.gcf().set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig("confusion_matrix_svm.png", dpi=150)

    # keep the model with the best macro F1 (external test if we have it)
    key = "external test" if ext is not None else "test split"
    best = table[table["data"] == key].sort_values("macro_f1").iloc[-1]["model"]
    joblib.dump(models[best], "model.joblib")
    print(f"\nSaved {best} to model.joblib, confusion matrix to confusion_matrix_svm.png")


if __name__ == "__main__":
    main()
