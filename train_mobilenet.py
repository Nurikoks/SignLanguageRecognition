"""Fine-tune MobileNetV2 (ImageNet weights) on the ASL Alphabet images and test it.

Usage:
  python train_mobilenet.py                                   # whole images
  python train_mobilenet.py --crop-hand --out mobilenet_hand.pt   # only the hand
  python train_mobilenet.py --eval-only --out mobilenet.pt    # test a saved model

First only the last layer is trained, then the whole network. The best epoch is chosen
on a validation part of the training set. The external test set is used only once, at
the end. The training images are frames of videos, so neighbours are almost the same
picture: validation takes whole blocks of frames, otherwise its score is too high.

With --crop-hand the network sees a square around the hand that MediaPipe found, not
the background. Images where no hand is found are left out of training and count as
mistakes in the test, the same as for SVM and Random Forest. The hand boxes are saved
in hand_boxes.csv, so MediaPipe runs over the images only once.
"""
import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

IMAGE_EXT = (".jpg", ".jpeg", ".png")
BLOCK = 50        # frames in a row that stay on the same side of the split
VAL_EVERY = 5     # every 5th block goes to validation (20%)
MARGIN = 1.4      # the crop is 1.4 times the size of the hand
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet


class Images(Dataset):
    """items: (path, label, box). box is None for the whole image."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, box = self.items[i]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if box is not None:
                image = image.crop(box)   # parts outside the image become black
            return self.transform(image), label


def list_images(folder):
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT)


def split_blocks(folders, per_class=None):
    train, val = [], []
    rng = random.Random(42)
    for label, folder in enumerate(folders):
        files = list_images(folder)
        if per_class and len(files) > per_class:
            files = rng.sample(files, per_class)
        for f in files:
            number = int(re.search(r"\d+", f.stem).group())
            (val if (number // BLOCK) % VAL_EVERY == 0 else train).append((f, label))
    return train, val


def square_box(points):
    """A square around the 21 landmarks (pixels), MARGIN times bigger than the hand."""
    x0, y0 = points[:, :2].min(axis=0)
    x1, y1 = points[:, :2].max(axis=0)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * MARGIN / 2
    return tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))


def hand_boxes(files, cache_path):
    """Box of the hand for every file (None if MediaPipe finds no hand), with a cache."""
    import cv2
    from features import create_detector, detect

    cache = {}
    if Path(cache_path).exists():
        with open(cache_path, newline="") as f:
            for row in csv.DictReader(f):
                cache[row["file"]] = (tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
                                      if row["found"] == "1" else None)
    todo = [f for f in files if f.as_posix() not in cache]
    if todo:
        print(f"Looking for hands in {len(todo)} images (saved to {cache_path})", flush=True)
        detector = create_detector()
        new_file = not Path(cache_path).exists()
        with open(cache_path, "a", newline="") as out:   # appended as we go: a crash loses little
            writer = csv.writer(out)
            if new_file:
                writer.writerow(["file", "found", "x0", "y0", "x1", "y1"])
            for i, f in enumerate(todo, 1):
                bgr = cv2.imread(str(f))
                points = None if bgr is None else detect(
                    detector, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[0]
                box = None if points is None else square_box(points)
                cache[f.as_posix()] = box
                writer.writerow([f.as_posix(), int(box is not None), *(box or ("", "", "", ""))])
                if i % 5000 == 0:
                    out.flush()
                    print(f"  {i}/{len(todo)}", flush=True)
    return [cache[f.as_posix()] for f in files]


def predict(model, loader, device):
    model.eval()
    pred = []
    with torch.inference_mode():
        for x, _ in loader:
            pred.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(pred).numpy() if pred else np.zeros(0, int)


def report(name, y, pred, labels):
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    print(f"{name:32s} n={len(y):5d}  accuracy={acc:.3f}  macro_f1={f1:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/asl_alphabet_train")
    parser.add_argument("--test", default="data/asl-alphabet-test")
    parser.add_argument("--crop-hand", action="store_true", help="train and test on the hand only")
    parser.add_argument("--boxes", default="hand_boxes.csv", help="cache of the hand boxes")
    parser.add_argument("--head-epochs", type=int, default=2, help="only the last layer")
    parser.add_argument("--epochs", type=int, default=8, help="the whole network")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=None, help="fewer images, for a quick try")
    parser.add_argument("--out", default="mobilenet.pt")
    parser.add_argument("--eval-only", action="store_true",
                        help="do not train, only test the model saved in --out")
    args = parser.parse_args()

    # imported here so the DataLoader workers do not have to load MediaPipe
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    from extract_features import find_class_folders
    from features import LETTERS

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folders = find_class_folders(args.train)
    test_folders = find_class_folders(args.test)
    if sorted(folders) != LETTERS or sorted(test_folders) != LETTERS:
        raise SystemExit("Need a folder for each of the 24 letters in train and test data")
    train_items, val_items = split_blocks([folders[l] for l in LETTERS], args.per_class)
    test_items = [(f, i) for i, l in enumerate(LETTERS) for f in list_images(test_folders[l])]

    # the test boxes are needed in both modes: results are also shown for "hand found" images
    test_boxes = hand_boxes([f for f, _ in test_items], args.boxes)
    found = np.array([b is not None for b in test_boxes])
    if args.crop_hand and not args.eval_only:
        boxes = hand_boxes([f for f, _ in train_items + val_items], args.boxes)
        n = len(train_items)
        train_items = [(f, l, b) for (f, l), b in zip(train_items, boxes[:n]) if b is not None]
        val_items = [(f, l, b) for (f, l), b in zip(val_items, boxes[n:]) if b is not None]
    else:
        train_items = [(f, l, None) for f, l in train_items]
        val_items = [(f, l, None) for f, l in val_items]
    test_items = [(f, l, b if args.crop_hand else None)
                  for (f, l), b in zip(test_items, test_boxes)]
    print(f"Device: {device}  Train: {len(train_items)}  Val: {len(val_items)}  "
          f"Test: {len(test_items)} (hand found in {found.sum()})")

    # the training photos look very alike, so the augmentation is strong
    scale = (0.8, 1.0) if args.crop_hand else (0.6, 1.0)   # do not cut fingers off the crop
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=scale, ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),                # a mirrored hand is the other hand, same letter
        T.RandomRotation(15),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25),
    ])
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(Images(train_items, train_tf), args.batch, shuffle=True,
                              drop_last=True, **options)
    val_loader = DataLoader(Images(val_items, eval_tf), args.batch * 2, **options)
    y_val = np.array([label for _, label, _ in val_items])
    y_test = np.array([label for _, label, _ in test_items])

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LETTERS))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_f1 = -1.0
    total = 0 if args.eval_only else args.head_epochs + args.epochs
    labels = list(range(len(LETTERS)))
    for epoch in range(total):
        if epoch == 0 or epoch == args.head_epochs:
            frozen = epoch < args.head_epochs
            for p in model.features.parameters():
                p.requires_grad = not frozen
            params = [p for p in model.parameters() if p.requires_grad]
            steps = args.head_epochs if frozen else args.epochs
            optimizer = torch.optim.AdamW(params, lr=1e-3 if frozen else 3e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, steps * len(train_loader))

        model.train()
        if epoch < args.head_epochs:
            model.features.eval()    # do not change BatchNorm while the network is frozen
        sum_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            sum_loss += loss.item()

        pred = predict(model, val_loader, device)
        val_acc = accuracy_score(y_val, pred)
        val_f1 = f1_score(y_val, pred, labels=labels, average="macro", zero_division=0)
        saved = val_f1 > best_f1
        if saved:
            best_f1 = val_f1
            torch.save({"state_dict": model.state_dict(), "letters": LETTERS,
                        "crop_hand": args.crop_hand}, args.out)
        print(f"Epoch {epoch + 1}/{total}  loss {sum_loss / len(train_loader):.3f}  "
              f"val accuracy {val_acc:.3f}  val macro F1 {val_f1:.3f}" + ("  saved" if saved else ""),
              flush=True)

    model.load_state_dict(torch.load(args.out, map_location=device)["state_dict"])
    # no worker processes here: each one loads the CUDA libraries again, and late in the run
    # Windows can run out of memory for them (WinError 1455). 720 images do not need them.
    usable = found if args.crop_hand else np.ones(len(test_items), bool)
    test_loader = DataLoader(Images([it for it, ok in zip(test_items, usable) if ok], eval_tf),
                             args.batch * 2)
    test_pred = np.full(len(test_items), -1)     # -1: no hand found, counts as a mistake
    test_pred[usable] = predict(model, test_loader, device)

    print()
    report("External test, all images", y_test, test_pred, labels)
    report("External test, hand found", y_test[found], test_pred[found], labels)
    print("\nPer-letter report (external test, all images):")
    print(classification_report(y_test, test_pred, labels=labels, target_names=LETTERS,
                                zero_division=0))

    matrix = f"confusion_matrix_{Path(args.out).stem}.png"
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, labels=labels,
                                            display_labels=LETTERS, cmap="Blues", colorbar=False)
    plt.title(f"MobileNetV2{' (hand crop)' if args.crop_hand else ''} confusion matrix "
              "(external test)")
    plt.gcf().set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(matrix, dpi=150)
    if not args.eval_only:
        print(f"Saved the best epoch to {args.out}")
    print(f"Saved the confusion matrix to {matrix}")


if __name__ == "__main__":
    main()
