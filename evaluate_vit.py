"""Test a ready ViT model for the ASL alphabet (no training) on the external test set.

Usage:
  python evaluate_vit.py
  python evaluate_vit.py --model path/to/downloaded/model

The model is dima806/asl_alphabet_image_detection from Hugging Face (ViT-Base, about
340 MB, downloaded on the first run). It knows 29 classes: A to Z, del, nothing, space.
We only have 24 letters, so the other 5 classes are never chosen.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torchvision import transforms as T
from transformers import ViTForImageClassification

from extract_features import find_class_folders
from features import LETTERS
from train_mobilenet import hand_boxes, list_images, report

MODEL = "dima806/asl_alphabet_image_detection"
REVISION = "54f2be1101086e06a81f0301275b0ffae64bd165"   # the version we tested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/asl-alphabet-test")
    parser.add_argument("--model", default=MODEL, help="Hugging Face name or a local folder")
    parser.add_argument("--boxes", default="hand_boxes.csv", help="cache of the hand boxes")
    args = parser.parse_args()

    folders = find_class_folders(args.test)
    items = [(f, i) for i, l in enumerate(LETTERS) for f in list_images(folders[l])]
    y = np.array([label for _, label in items])
    found = np.array([b is not None for b in hand_boxes([f for f, _ in items], args.boxes)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(
        args.model, revision=None if args.model != MODEL else REVISION,
        use_safetensors=True).to(device).eval()
    # model class id for each of our letters
    ids = {label: i for i, label in model.config.id2label.items()}
    letter_ids = torch.tensor([ids[l] for l in LETTERS], device=device)

    # same as the model's preprocessor_config.json: 224x224, mean 0.5, std 0.5
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    pred = []
    with torch.inference_mode():
        for s in range(0, len(items), 32):
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f, _ in items[s:s + 32]])
            logits = model(pixel_values=x.to(device)).logits
            pred.append(logits[:, letter_ids].argmax(1).cpu())   # only our 24 letters
    pred = torch.cat(pred).numpy()

    labels = list(range(len(LETTERS)))
    print(f"Device: {device}  Test: {len(items)} (hand found in {found.sum()})\n")
    report("External test, all images", y, pred, labels)
    report("External test, hand found", y[found], pred[found], labels)
    print("\nPer-letter report (external test, all images):")
    print(classification_report(y, pred, labels=labels, target_names=LETTERS, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y, pred, labels=labels, display_labels=LETTERS,
                                            cmap="Blues", colorbar=False)
    plt.title("ViT (ready model) confusion matrix (external test)")
    plt.gcf().set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig("confusion_matrix_vit.png", dpi=150)
    print("Saved the confusion matrix to confusion_matrix_vit.png")


if __name__ == "__main__":
    main()
