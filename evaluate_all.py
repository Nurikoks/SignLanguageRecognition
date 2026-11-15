"""Compute every number and figure of the experiments from the saved predictions.

Usage:
  python evaluate_all.py                 # letters, predictions/*.csv
  python evaluate_all.py --task words    # words, predictions_words/*.csv

Letters: run train.py, train_mobilenet.py (both models) and evaluate_vit.py first. Each
of them saves its predictions on the 720 external test images.
Words: run train_islr.py first. It saves the predictions of 4 models on the clips of
the test people.
This script only reads those files, so all models are compared on exactly the same data.

Output (for words the names start with words_):
  figures/metrics.md             all tables (main results, McNemar, per-class F1, mistakes)
  figures/confusion_<model>.png  confusion matrix of each model
  figures/model_comparison.png   accuracy of all models with 95% confidence intervals
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from features import LETTERS
from results import FIG_DIR, PRED_DIR

# file name -> name in the tables, in the order of the tables
MODELS = {
    "svm": "SVM",
    "rf": "Random Forest",
    "vit": "ViT (ready model)",
    "mobilenet_hand": "MobileNetV2, hand crop",
    "mobilenet": "MobileNetV2, whole image",
}
WORD_MODELS = {"svm": "SVM", "rf": "Random Forest", "cnn": "1D-CNN", "gru": "GRU"}
WORD_PRED_DIR = Path("predictions_words")
BOOTSTRAP = 2000


def load(models, pred_dir):
    preds, first = {}, None
    for key in models:
        path = pred_dir / f"{key}.csv"
        if not path.exists():
            print(f"Missing {path}, this model is left out")
            continue
        df = pd.read_csv(path, keep_default_na=False, dtype={"pred": str, "true": str})
        if first is None:
            first = df
        elif df["file"].tolist() != first["file"].tolist():
            raise SystemExit(f"{path} has other test data than the first file")
        preds[key] = df["pred"].to_numpy()
    if not preds:
        raise SystemExit(f"No predictions in {pred_dir}, run the model scripts first")
    return first, first["true"].to_numpy(), preds


def macro(true, pred, labels):
    p, r, f, _ = precision_recall_fscore_support(true, pred, labels=labels, average="macro",
                                                 zero_division=0)
    return p, r, f


def bootstrap_ci(correct, seed=42):
    """95% interval of the accuracy: resample the test examples with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), (BOOTSTRAP, len(correct)))
    acc = correct[idx].mean(axis=1)
    return np.percentile(acc, 2.5), np.percentile(acc, 97.5)


def mcnemar(a_right, b_right):
    """Exact McNemar test: only the examples where exactly one model is right count."""
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def per_class_f1(true, preds, labels, names, title):
    f1 = {k: precision_recall_fscore_support(true, p, labels=labels, zero_division=0)[2]
          for k, p in preds.items()}
    rows = [[c] + [f"{f1[k][i]:.2f}" for k in preds] for i, c in enumerate(labels)]
    return f"\n## {title}\n\n" + table(["Class"] + [names[k] for k in preds], rows)


def top_mistakes(true, preds, names, title):
    rows = []
    for k, pred in preds.items():
        wrong = pd.Series([f"{t} as {p or 'no answer'}" for t, p in zip(true, pred) if t != p])
        rows.append([names[k]] + [f"{m} ({c})" for m, c in wrong.value_counts().head(5).items()])
    return f"\n## {title}\n\n" + table(["Model", "1st", "2nd", "3rd", "4th", "5th"], rows)


def plot_confusion(cm, xlabels, ylabels, title, path, size=(9, 8.5), xlabel="predicted"):
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(cm, cmap="Blues")
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    long = max(len(x) for x in xlabels) > 1
    ax.set_xticks(range(cm.shape[1]), xlabels, rotation=90 if long else 0)
    ax.set_yticks(range(cm.shape[0]), ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(comparison, title, path, second=None):
    names = [c[0] for c in comparison]
    acc, lo, hi = (np.array([c[i] for c in comparison]) for i in (1, 2, 3))
    x = np.arange(len(names))
    w = 0.4 if second else 0.6
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - (0.2 if second else 0), acc, w, yerr=[acc - lo, hi - acc], capsize=4,
           label=comparison[0][4] if second else None)
    if second:
        ax.bar(x + 0.2, [c[5] for c in comparison], w, label=second)
        ax.legend()
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def letters():
    df, true, preds = load(MODELS, PRED_DIR)
    files = df["file"].tolist()
    # "hand found": the images where the landmark models got a hand (MediaPipe)
    found = np.isin(files, pd.read_csv("test.csv")["file"])
    n, n_found = len(files), int(found.sum())
    right = {k: p == true for k, p in preds.items()}
    out = ["# Letter results (generated by evaluate_all.py)\n",
           f"External test set: {n} images, MediaPipe finds a hand in {n_found}. "
           "A model with no answer (no hand) counts as a mistake in the 'all' columns.\n"]

    rows, comparison = [], []
    for k, pred in preds.items():
        lo, hi = bootstrap_ci(right[k])
        p, r, f = macro(true, pred, LETTERS)
        _, _, f_found = macro(true[found], pred[found], LETTERS)
        rows.append([MODELS[k], f"{right[k].mean():.1%} ({right[k].sum()})",
                     f"{lo:.1%} to {hi:.1%}", f"{right[k][found].mean():.1%}",
                     f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", f"{f_found:.3f}",
                     int((pred != "").sum())])
        comparison.append((MODELS[k], right[k].mean(), lo, hi, f"all {n} images (95% CI)",
                           right[k][found].mean()))
    out.append("## Main results (external test)\n")
    out.append(table(["Model", f"Accuracy, all {n}", "95% CI (bootstrap)",
                      f"Accuracy, hand found ({n_found})", "Macro precision, all",
                      "Macro recall, all", f"Macro F1, all {n}", "Macro F1, hand found",
                      "Images answered"], rows))

    rows = []
    for a, b in combinations(preds, 2):
        for name, mask in ((f"all {n}", np.ones(n, bool)), ("hand found", found)):
            only_a, only_b, p = mcnemar(right[a][mask], right[b][mask])
            rows.append([MODELS[a], MODELS[b], name, only_a, only_b, fmt_p(p)])
    out.append("\n## McNemar test (exact, two-sided)\n")
    out.append(table(["Model A", "Model B", "Images", "Only A right", "Only B right", "p-value"],
                     rows))
    out.append(per_class_f1(true, preds, LETTERS, MODELS,
                            f"F1 for each letter (external test, all {n} images)"))
    out.append(top_mistakes(true, preds, MODELS,
                            "Most common mistakes (true letter as predicted, count of 30)"))

    FIG_DIR.mkdir(exist_ok=True)
    (FIG_DIR / "metrics.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))

    labels = LETTERS + ["none"]
    for k, pred in preds.items():
        cm = confusion_matrix(true, np.where(pred == "", "none", pred), labels=labels)[:len(LETTERS)]
        if cm[:, -1].sum() == 0:
            cm = cm[:, :-1]
        plot_confusion(cm, labels[:cm.shape[1]], LETTERS,
                       f"{MODELS[k]}: confusion matrix, external test "
                       f"(accuracy {right[k].mean():.1%}, n = {n})", FIG_DIR / f"confusion_{k}.png",
                       xlabel="predicted (none = no hand found, no answer)")
    plot_comparison(comparison, "Accuracy on the external test set (ASL Alphabet Test)",
                    FIG_DIR / "model_comparison.png", second=f"hand found ({n_found})")
    print(f"\nSaved {FIG_DIR}/metrics.md and the figures")


def words():
    from words import WORDS
    df, true, preds = load(WORD_MODELS, WORD_PRED_DIR)
    person = df["person"].to_numpy()
    people = sorted(set(person))
    n = len(df)
    right = {k: p == true for k, p in preds.items()}
    out = ["# Word results (generated by evaluate_all.py --task words)\n",
           f"Test set: {n} clips of {len(WORDS)} words by {len(people)} people that the "
           "models never saw in training (Kaggle ISLR).\n"]

    rows, comparison = [], []
    for k, pred in preds.items():
        lo, hi = bootstrap_ci(right[k])
        p, r, f = macro(true, pred, WORDS)
        rows.append([WORD_MODELS[k], f"{right[k].mean():.1%} ({right[k].sum()})",
                     f"{lo:.1%} to {hi:.1%}", f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"])
        comparison.append((WORD_MODELS[k], right[k].mean(), lo, hi))
    out.append("## Main results (test people)\n")
    out.append(table(["Model", f"Accuracy ({n} clips)", "95% CI (bootstrap)", "Macro precision",
                      "Macro recall", "Macro F1"], rows))

    rows = [[WORD_MODELS[k]] + [f"{right[k][person == q].mean():.1%}" for q in people]
            for k in preds]
    out.append("\n## Accuracy for each test person\n")
    out.append(table(["Model"] + [f"person {q} ({(person == q).sum()} clips)" for q in people],
                     rows))

    rows = []
    for a, b in combinations(preds, 2):
        only_a, only_b, p = mcnemar(right[a], right[b])
        rows.append([WORD_MODELS[a], WORD_MODELS[b], only_a, only_b, fmt_p(p)])
    out.append("\n## McNemar test (exact, two-sided)\n")
    out.append(table(["Model A", "Model B", "Only A right", "Only B right", "p-value"], rows))
    out.append(per_class_f1(true, preds, WORDS, WORD_MODELS, "F1 for each word"))
    out.append(top_mistakes(true, preds, WORD_MODELS,
                            "Most common mistakes (true word as predicted, count)"))

    FIG_DIR.mkdir(exist_ok=True)
    (FIG_DIR / "words_metrics.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    for k, pred in preds.items():
        plot_confusion(confusion_matrix(true, pred, labels=WORDS), WORDS, WORDS,
                       f"{WORD_MODELS[k]}: confusion matrix, test people "
                       f"(accuracy {right[k].mean():.1%}, n = {n})",
                       FIG_DIR / f"words_confusion_{k}.png", size=(10, 9.5))
    plot_comparison(comparison, f"Word accuracy for {len(people)} unseen people (95% CI)",
                    FIG_DIR / "words_model_comparison.png")
    print(f"\nSaved {FIG_DIR}/words_metrics.md and the figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["letters", "words"], default="letters")
    (words if parser.parse_args().task == "words" else letters)()


if __name__ == "__main__":
    main()
