"""Train the word models on the Kaggle ISLR clips (signs with movement).

Usage:
  python islr_prepare.py       # first: take the clips of the words in words.py
  python train_islr.py

The 21 people of the dataset are split: 14 for training, 3 for validation (to choose
the epoch) and 4 for the test. So the test shows how the models work for people they
have never seen. Models:
  SVM and Random Forest on a summary of the clip (word_features.summary)
  1D-CNN and GRU (PyTorch) on the whole sequence, STEPS x FRAME_DIM
The neural networks see a new random augmentation (stretch, rotation, scale, shift,
cut at the start and the end) in every epoch. Their loss and accuracy per epoch are saved
to figures/history_words_<model>.csv and figures/learning_curve_words_<model>.png.
Test predictions go to predictions_words/<model>.csv (evaluate_all.py --task words).
The best model on validation is saved to words_model.pt for the webcam demo.
"""
import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from results import FIG_DIR
from word_features import FRAME_DIM, STEPS, random_augment, sequence, summary
from words import ISLR_DIR, WORDS

PRED_DIR = Path("predictions_words")
COPIES = 8          # augmented copies of every training clip, made once
SEED = 42


def split_people(people, seed=SEED):
    rng = np.random.default_rng(seed)
    people = rng.permutation(sorted(people))
    return set(people[:4]), set(people[4:7]), set(people[7:])    # test, val, train


def augmented(clip, rng):
    T = len(clip)
    cut = int(T * 0.1)
    a, b = rng.integers(0, cut + 1), T - rng.integers(0, cut + 1)
    return sequence(clip[a:b] if b - a >= 4 else clip, augment=random_augment(rng))


def build(meta, copies, seed=SEED):
    """Sequences of all clips: X (n x STEPS x FRAME_DIM), and copies x augmented."""
    rng = np.random.default_rng(seed)
    X, keep, aug = [], [], []
    for i, sid in enumerate(meta["sequence_id"]):
        clip = np.load(ISLR_DIR / "clips" / f"{sid}.npy").astype(np.float32)
        s = sequence(clip)
        if s is None:
            continue
        X.append(s)
        keep.append(i)
        if copies:
            copies_of = [augmented(clip, rng) for _ in range(copies)]
            aug.append([s if a is None else a for a in copies_of])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(meta)} clips", flush=True)
    X = np.stack(X)
    A = np.stack([np.stack(a) for a in aug]).astype(np.float16) if copies else None
    return X, meta.iloc[keep].reset_index(drop=True), A


class CNN(nn.Module):
    def __init__(self, n_classes, width=192, dropout=0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.BatchNorm1d(cout),
                                 nn.GELU(), nn.Dropout(dropout))
        self.net = nn.Sequential(block(FRAME_DIM, width), block(width, width), nn.MaxPool1d(2),
                                 block(width, width), block(width, width))
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, x):                 # x: batch x STEPS x FRAME_DIM
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


class GRU(nn.Module):
    def __init__(self, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FRAME_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4 * hidden, n_classes)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.head(self.drop(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)))


NETS = {"cnn": CNN, "gru": GRU}
NAMES = {"svm": "SVM", "rf": "Random Forest", "cnn": "1D-CNN", "gru": "GRU"}


def plot_history(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    a.plot(e, [h["train_loss"] for h in history], label="training")
    a.plot(e, [h["val_loss"] for h in history], label="validation (other people)")
    b.plot(e, [h["train_acc"] for h in history], label="training")
    b.plot(e, [h["val_acc"] for h in history], label="validation (other people)")
    best = max(history, key=lambda h: h["val_acc"])["epoch"]
    for ax, name in ((a, "loss (cross-entropy, label smoothing 0.1)"), (b, "accuracy")):
        ax.axvline(best, color="gray", ls="--", lw=1, label=f"best epoch ({best})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_net(key, A_train, y_train, X_val, y_val, args, device):
    torch.manual_seed(SEED)
    model = NETS[key](len(WORDS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(y_train) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xv, yv = torch.tensor(X_val, device=device), torch.tensor(y_val, device=device)
    yt_all = torch.tensor(y_train, device=device)
    A = torch.tensor(A_train, device=device)     # n x COPIES x STEPS x DIM, float16
    rng = np.random.default_rng(SEED)
    history, best, path = [], -1.0, Path(f"words_{key}.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.tensor(rng.permutation(len(y_train)), device=device)
        copy = torch.tensor(rng.integers(0, A.shape[1], len(y_train)), device=device)
        total, correct, loss_sum = 0, 0, 0.0
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            x, y = A[idx, copy[idx]].float(), yt_all[idx]
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        model.eval()
        with torch.inference_mode():
            lv = model(Xv)
            val_loss = loss_fn(lv, yv).item()
            val_acc = (lv.argmax(1) == yv).float().mean().item()
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best:
            best = val_acc
            torch.save({"model": key, "state_dict": model.state_dict(), "words": WORDS,
                        "steps": STEPS, "frame_dim": FRAME_DIM}, path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {NAMES[key]} epoch {epoch}: train loss {loss_sum / total:.3f} acc "
                  f"{correct / total:.3f}  val loss {val_loss:.3f} acc {val_acc:.3f}", flush=True)
    with open(FIG_DIR / f"history_words_{key}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, history[0].keys())
        w.writeheader()
        w.writerows(history)
    plot_history(history, FIG_DIR / f"learning_curve_words_{key}.png",
                 f"{NAMES[key]} training on ISLR words ({len(WORDS)} words)")
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    model.eval()
    return model, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--clips", default=str(ISLR_DIR / "clips.csv"))
    args = parser.parse_args()

    meta = pd.read_csv(args.clips)
    meta = meta[meta["sign"].isin(WORDS)].reset_index(drop=True)
    test_p, val_p, train_p = split_people(meta["participant_id"].unique())
    cache = ISLR_DIR / f"sequences_{Path(args.clips).stem}.npz"
    ids = meta["sequence_id"].to_numpy()
    d = np.load(cache) if cache.exists() else None
    if d is not None and np.array_equal(d["ids"], ids):     # same clips as last time
        X, A = d["X"], d["A"]
        meta = meta.iloc[d["keep"]].reset_index(drop=True)
    else:
        print(f"Making sequences of {len(meta)} clips ({COPIES} augmented copies each)")
        start = time.time()
        meta["row"] = np.arange(len(meta))
        X, meta, A = build(meta, COPIES)
        np.savez(cache, X=X, A=A, keep=meta["row"].to_numpy(), ids=ids)
        print(f"  done in {time.time() - start:.0f} s")
    y = meta["sign"].map({w: i for i, w in enumerate(WORDS)}).to_numpy()
    part = meta["participant_id"]
    tr, va, te = part.isin(train_p).to_numpy(), part.isin(val_p).to_numpy(), part.isin(test_p).to_numpy()
    print(f"Clips: train {tr.sum()} ({len(train_p)} people), validation {va.sum()} "
          f"({len(val_p)} people), test {te.sum()} ({len(test_p)} people)")
    print(f"Test people: {sorted(int(p) for p in test_p)}")

    FIG_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    test_true = [WORDS[i] for i in y[te]]
    val_scores = {}

    def save(key, pred):
        pd.DataFrame({"file": meta.loc[te, "sequence_id"].to_numpy(), "person": part[te].to_numpy(),
                      "true": test_true, "pred": [WORDS[i] for i in pred]}).to_csv(
            PRED_DIR / f"{key}.csv", index=False)
        print(f"{NAMES[key]:14s} test accuracy {accuracy_score(y[te], pred):.3f}  "
              f"macro F1 {f1_score(y[te], pred, average='macro'):.3f}", flush=True)

    # classic models: summary features, original clips + 2 augmented copies
    S = np.stack([summary(s) for s in X])
    S_aug = np.stack([summary(s) for s in
                      A[tr, :2].astype(np.float32).reshape(-1, STEPS, FRAME_DIM)])
    S_train = np.concatenate([S[tr], S_aug])
    y_train = np.concatenate([y[tr], np.repeat(y[tr], 2)])
    classic = {"svm": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True,
                                                          random_state=SEED)),
               "rf": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)}
    for key, model in classic.items():
        start = time.time()
        model.fit(S_train, y_train)
        val_scores[key] = accuracy_score(y[va], model.predict(S[va]))
        print(f"{NAMES[key]:14s} validation accuracy {val_scores[key]:.3f} ({time.time() - start:.0f} s)")
        joblib.dump(model, f"words_{key}.joblib")
        save(key, model.predict(S[te]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in NETS:
        start = time.time()
        model, val_scores[key] = train_net(key, A[tr], y[tr], X[va], y[va], args, device)
        print(f"{NAMES[key]:14s} best validation accuracy {val_scores[key]:.3f} "
              f"({time.time() - start:.0f} s)")
        with torch.inference_mode():
            save(key, model(torch.tensor(X[te], device=device)).argmax(1).cpu().numpy())

    best = max(NETS, key=lambda k: val_scores[k])
    Path("words_model.pt").write_bytes(Path(f"words_{best}.pt").read_bytes())
    print("\nValidation accuracy: " + ", ".join(f"{NAMES[k]} {v:.3f}" for k, v in val_scores.items()))
    print(f"{NAMES[best]} is the best network on validation, copied to words_model.pt for the demo")


if __name__ == "__main__":
    main()
