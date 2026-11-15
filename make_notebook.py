"""Build SignVision.ipynb in the course template format, then run it:
  python make_notebook.py
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 SignVision.ipynb
  python make_notebook.py --clean      # remove the C++ log lines of MediaPipe from the outputs
  python make_notebook.py --markdown   # only update the text cells of the executed notebook
  python make_notebook.py --pdf        # SignVision.pdf (HTML printed by Edge)

The first cell is the README of the project. Then every step has a heading, a short
explanation and the code with its real output.
"""
import re
import sys

import nbformat as nbf

# log lines that MediaPipe / TensorFlow Lite print from C++, not output of our code
LOG_LINE = re.compile(r"^([WIE]\d{4} [\d:.]+ +\d+ \S+\] .*|INFO: Created TensorFlow Lite XNNPACK delegate for CPU\.|"
                      r"WARNING: Logging before InitGoogle\(\) is written to STDERR)\n?$")


def clean(path="SignVision.ipynb"):
    nb = nbf.read(path, 4)
    removed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        outputs = []
        for out in cell.outputs:
            if out.output_type == "stream":
                lines = out.text.splitlines(keepends=True)
                keep = [l for l in lines if not LOG_LINE.match(l)]
                removed += len(lines) - len(keep)
                out.text = "".join(keep)
                if not out.text.strip():
                    continue
            outputs.append(out)
        cell.outputs = outputs
    nbf.write(nb, path)
    print(f"Removed {removed} log lines from {path}")


EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
WRAP_CSS = "<style>pre, .jp-OutputArea-output pre { white-space: pre-wrap !important; word-break: break-word; }</style>"


def to_pdf(path="SignVision.ipynb"):
    """HTML with wrapped long lines, printed to PDF by Edge (no LaTeX needed)."""
    import subprocess
    from pathlib import Path
    html = Path(path).with_suffix(".html").resolve()
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", path], check=True)
    html.write_text(html.read_text(encoding="utf-8").replace("</head>", WRAP_CSS + "</head>", 1),
                    encoding="utf-8")
    pdf = html.with_suffix(".pdf")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    print(f"Saved {pdf}")


if "--clean" in sys.argv:
    clean()
    sys.exit()
if "--pdf" in sys.argv:
    to_pdf()
    sys.exit()

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def step(title, text, source=None):
    md(f"### {title}\n{text.strip()}")
    if source:
        code(source)


md("""
# Sign Language Recognition Project (SignVision)

## Authors
- Nurassyl Kaziyev
- Alua Muratova

## Project Description
This project recognizes American Sign Language (ASL) from a camera. It has two tasks:
24 static letters (A to Y without J and Z) from images, and 25 words with movement (hello,
thank you, water, ...) from video. We compare classical machine learning on hand landmarks
with neural networks and a ready transformer model, and test all models on people that they
have never seen. A webcam application shows the models working live.

## Technologies Used
- Python
- MediaPipe (hand and pose landmarks)
- Scikit-learn (SVM, Random Forest)
- PyTorch and torchvision (MobileNetV2, 1D-CNN, GRU)
- Hugging Face Transformers (ViT)
- OpenCV (webcam application)

## Datasets
- ASL Alphabet (training of letters): https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- ASL Alphabet Test (test of letters): https://www.kaggle.com/datasets/danrasband/asl-alphabet-test
- Google Isolated Sign Language Recognition (words): https://www.kaggle.com/competitions/asl-signs

## How to Run
1. Install the libraries: `pip install -r requirements.txt` (and PyTorch with CUDA for the networks).
2. Download the datasets into `data/` and the MediaPipe models (see `README.md` in the repository).
3. Extract the hand landmarks: `extract_features.py`.
4. Train the letter models: `train.py`, `train_mobilenet.py`, test the ready model: `evaluate_vit.py`.
5. Prepare and train the word models: `islr_prepare.py`, `train_islr.py`.
6. Compute all metrics and figures: `evaluate_all.py`.
7. (Optional) Run the webcam application: `webcam_demo.py`.

## Models Used
- Support Vector Machine (letters and words)
- Random Forest (letters and words)
- MobileNetV2, fine-tuned from ImageNet (letters)
- Vision Transformer (ViT), ready model from Hugging Face (letters)
- 1D Convolutional Neural Network (words)
- GRU recurrent neural network (words)
""")

md("""
## Notes on this notebook
The notebook runs the project scripts and shows their real output. MobileNetV2 is not trained
again here, because one model takes about 45 minutes on a GPU. We test its saved weights and
show the loss and accuracy that were saved during its training.
The C++ log lines of MediaPipe (for example "Feedback manager requires a model with a single
signature") are removed from the printed output, everything else is shown as it was printed.
""")

step("Importing Libraries", "The main libraries for data, plots and models are imported.", """
import platform
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image, Markdown, display


def show(path, width=900):
    display(Image(filename=str(path), width=width))


def section(md_file, title):
    \"\"\"Show one table (## title) from a results file written by evaluate_all.py.\"\"\"
    text = Path(md_file).read_text(encoding="utf-8")
    display(Markdown(text.split("## " + title, 1)[1].split("\\n## ", 1)[0]))
""")

step("Checking GPU Availability", "The neural networks are trained on a GPU with CUDA.", """
print("Python", platform.python_version(), "| PyTorch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
""")

md("# Part 1. Static Letters")

step("Loading the Letter Dataset", """
The training set has 3,000 images for each letter. The images are frames of videos with the
same background. The external test set has 30 images for each letter with other people and
backgrounds. We use the 24 static letters, because J and Z need movement.
""", """
from extract_features import find_class_folders
from features import LETTERS
from results import is_val, list_images, test_images

train_folders = find_class_folders("data/asl_alphabet_train")
n_train = {l: len(list_images(train_folders[l])) for l in LETTERS}
test = test_images()
print("Letters:", " ".join(LETTERS))
print("Training images:", sum(n_train.values()), "| per letter:", set(n_train.values()))
print("External test images:", len(test), "| per letter:", set(pd.Series([l for _, l in test]).value_counts()))
""")

step("Showing Example Images", "The top row shows training images, the bottom row shows test images of the same letters.", """
fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for ax, letter in zip(axes[0], "ABCLNMSV"):
    ax.imshow(plt.imread(list_images(train_folders[letter])[100])); ax.set_title("train " + letter); ax.axis("off")
for ax, letter in zip(axes[1], "ABCLNMSV"):
    path = next(p for p, l in test if l == letter)
    ax.imshow(plt.imread(path)); ax.set_title("test " + letter); ax.axis("off")
plt.tight_layout(); plt.show()
""")

step("Splitting Training and Validation Data", """
Neighbouring video frames look almost the same. If we split them at random, the same picture
would be in training and validation. So we split them in blocks of 50 frames in a row, and
every fifth block is used for validation (`results.py`).
""", """
files = list_images(train_folders["A"])
val = [is_val(f) for f in files]
print("Letter A, frames 0-299 (V = validation, . = training):")
print("".join("V" if v else "." for v in val[:300]))
print("Validation share:", round(np.mean(val), 3))
""")

step("Extracting Hand Landmarks with MediaPipe", """
MediaPipe finds 21 points of the hand. We move the wrist to (0, 0), divide by the hand size and
mirror the left hand. This gives 63 numbers per image that do not depend on the position and
size of the hand or on which hand is used (`features.py`).
""", """
from features import create_detector, detect, draw_hand, normalize

detector = create_detector()
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (path, letter) in zip(axes, [test[0], test[90], test[300], test[600]]):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    points, side = detect(detector, rgb)
    draw_hand(rgb, points, (255, 120, 0))
    ax.imshow(rgb); ax.set_title(f"{letter}, {side} hand"); ax.axis("off")
plt.tight_layout(); plt.show()
print("First 9 of the 63 features of the last image:", np.round(normalize(points, side)[:9], 3))
""")

step("Extracting Features of the Test Images", """
The landmarks of the 720 test images are saved to `test.csv`. The 72,000 training images were
processed by the same script before (`train.csv`), which takes much longer.
""", """
!python extract_features.py --data data/asl-alphabet-test --out test.csv
""")

step("Loading the Feature Files", "Each row has 63 features, the letter and the image file. Images without a found hand are not saved.", """
train_csv, test_csv = pd.read_csv("train.csv"), pd.read_csv("test.csv")
print("train.csv:", train_csv.shape, "| a hand was found in", f"{len(train_csv) / 72000:.1%}", "of the training images")
print("test.csv: ", test_csv.shape, "| a hand was found in", len(test_csv), "of 720 test images")
train_csv.head(3)
""")

step("Training SVM and Random Forest", """
`train.py` first makes a learning curve (both models trained on 50 to 3,000 images per letter)
and then trains the final models on all training images. SVM uses an RBF kernel (C = 10) and
Random Forest has 300 trees. The model that is better on validation is used in the webcam
application. The external test set is not used to choose anything.
""", """
!python train.py
""")

step("Learning Curve", """
Accuracy on the training part, on validation (same dataset) and on the external test (new people)
for a growing number of training images.
""", """
show("figures/learning_curve_landmarks.png")
pd.read_csv("figures/learning_curve_landmarks.csv").round(3)
""")

step("MobileNetV2 Training History (Loss and Accuracy)", """
MobileNetV2 starts from ImageNet weights. The first 2 epochs train only the last layer (left of
the dashed line), then 8 epochs train the whole network (AdamW, cosine learning rate, label
smoothing 0.1). One model sees a square around the hand, the other the whole image.
""", """
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for row, (stem, name) in enumerate([("mobilenet_hand", "hand crop"), ("mobilenet", "whole image")]):
    h = pd.read_csv(f"figures/history_{stem}.csv")
    axes[row, 0].plot(h.epoch, h.train_loss, "o-", label="train"); axes[row, 0].plot(h.epoch, h.val_loss, "o-", label="validation")
    axes[row, 1].plot(h.epoch, h.train_acc, "o-", label="train"); axes[row, 1].plot(h.epoch, h.val_acc, "o-", label="validation")
    axes[row, 0].set_title(f"MobileNetV2 {name}: loss"); axes[row, 1].set_title(f"MobileNetV2 {name}: accuracy")
    for ax in axes[row]:
        ax.axvline(h.head_epochs[0] + 0.5, color="gray", ls="--"); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
pd.read_csv("figures/history_mobilenet_hand.csv").round(4)
""")

step("Testing MobileNetV2 (Hand Crop)", "The saved best epoch is tested on the external test set.", """
!python train_mobilenet.py --crop-hand --eval-only --out mobilenet_hand.pt
""")

step("Testing MobileNetV2 (Whole Image)", "The same for the model that sees the whole image.", """
!python train_mobilenet.py --eval-only --out mobilenet.pt
""")

step("Testing the Ready ViT Model", """
`dima806/asl_alphabet_image_detection` is a Vision Transformer from Hugging Face that its author
trained on the ASL Alphabet dataset. We do not train it, we only use its 24 letter outputs.
""", """
!python evaluate_vit.py
""")

step("Evaluating All Letter Models", """
`evaluate_all.py` reads the saved predictions of all five models on the same 720 test images.
A model that gives no answer (no hand found) counts as a mistake.
""", """
!python evaluate_all.py
""")

step("Letter Results", "Accuracy with a 95% bootstrap interval, and macro precision, recall and F1.", """
section("figures/metrics.md", "Main results (external test)")
show("figures/model_comparison.png", 800)
""")

step("Statistical Significance (McNemar Test)", """
The exact McNemar test compares two models on the same images. A p-value below 0.05 means the
difference is probably not random.
""", """
section("figures/metrics.md", "McNemar test (exact, two-sided)")
""")

step("F1-score for Each Letter and Common Mistakes", "The F1-score of every letter shows which signs are hard. The second table lists the five most common mistakes of each model.", """
section("figures/metrics.md", "F1 for each letter (external test, all 720 images)")
section("figures/metrics.md", "Most common mistakes (true letter as predicted, count of 30)")
""")

step("Confusion Matrix", "Confusion matrices of the best landmark model, the ready ViT and MobileNetV2 on the external test.", """
for name in ["svm", "vit", "mobilenet_hand"]:
    show(f"figures/confusion_{name}.png", 650)
""")

step("Testing the Letter Model on Example Images", "The saved SVM model predicts some external test images.", """
svm = joblib.load("svm.joblib")
fig, axes = plt.subplots(1, 6, figsize=(16, 3.4))
for ax, (path, letter) in zip(axes, test[5::121]):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    points, side = detect(detector, rgb)
    if points is None:
        title = f"true {letter}, no hand"
    else:
        probs = svm.predict_proba([normalize(points, side)])[0]
        title = f"true {letter}, SVM {svm.classes_[probs.argmax()]} ({probs.max():.0%})"
    ax.imshow(rgb); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
""")

md("# Part 2. Words with Movement")

step("Loading the Word Dataset", """
The Kaggle ISLR dataset has 94,477 short videos of 250 ASL signs by 21 people, given as MediaPipe
landmarks for every frame. We chose 25 everyday words (`words.py`) and kept only the hands and
the pose of their clips (`islr_prepare.py`).
""", """
from words import WORDS
clips = pd.read_csv("data/islr/clips.csv")
print(len(WORDS), "words:", ", ".join(WORDS))
print("Clips:", len(clips), "| people:", clips.participant_id.nunique())
lengths = [len(np.load(f"data/islr/clips/{s}.npy", mmap_mode="r")) for s in clips.sequence_id]
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
clips.sign.value_counts().plot.bar(ax=axes[0], title="clips per word")
axes[1].hist(lengths, bins=60, range=(0, 200)); axes[1].set_title("clip length (frames)")
plt.tight_layout(); plt.show()
print("Median clip length:", int(np.median(lengths)), "frames")
""")

step("Splitting the Dataset by Person", """
The clips are split by person: 14 people for training, 3 for validation and 4 for the test.
So the test shows how the models work for people they have never seen.
""", """
from train_islr import split_people
test_p, val_p, train_p = split_people(clips.participant_id.unique())
part = clips.participant_id.map(lambda p: "test" if p in test_p else "validation" if p in val_p else "train")
pd.DataFrame({"people": clips.groupby(part).participant_id.nunique(), "clips": part.value_counts()})
""")

step("Features of a Word Clip", """
For each frame we keep both hands (wrist position, hand shape, "hand seen" flag) and 7 upper body
points, measured from the middle of the shoulders in shoulder widths. Left-handed signs are
mirrored and every clip is resampled to 32 steps (`word_features.py`). Below: one clip of
"hello" and the 32 x 104 sequence that the networks get.
""", """
from word_features import sequence
sid = clips[clips.sign == "hello"].sequence_id.iloc[3]
clip = np.load(f"data/islr/clips/{sid}.npy").astype(np.float32)
seq = sequence(clip)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for start, color, name in ((0, "tab:orange", "hand 1"), (21, "tab:blue", "hand 2")):
    axes[0].plot(clip[:, start, 0], -clip[:, start, 1], ".-", color=color, label=name + " wrist")
axes[0].plot(clip[:, 42 + 11, 0], -clip[:, 42 + 11, 1], "k.", label="shoulders")
axes[0].plot(clip[:, 42 + 12, 0], -clip[:, 42 + 12, 1], "k.")
axes[0].set_title("hello: wrist paths in the image"); axes[0].legend(); axes[0].set_aspect("equal")
axes[1].imshow(seq.T, aspect="auto", cmap="coolwarm"); axes[1].set_title(f"model input {seq.shape} (steps x features)")
axes[1].set_xlabel("time step"); axes[1].set_ylabel("feature")
plt.tight_layout(); plt.show()
""")

step("Training the Word Models", """
SVM and Random Forest get a summary of the clip (1,040 numbers). The 1D-CNN and the GRU get the
whole 32 x 104 sequence and are trained for 80 epochs with random augmentation (AdamW, one-cycle
learning rate, label smoothing 0.1). The epoch with the best validation accuracy is kept.
""", """
!python train_islr.py
""")

step("Word Models Training History (Loss and Accuracy)", "Loss and accuracy of the 1D-CNN and the GRU for each of the 80 epochs, on the training and the validation people.", """
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for row, key in enumerate(["cnn", "gru"]):
    h = pd.read_csv(f"figures/history_words_{key}.csv")
    axes[row, 0].plot(h.epoch, h.train_loss, label="train"); axes[row, 0].plot(h.epoch, h.val_loss, label="validation")
    axes[row, 1].plot(h.epoch, h.train_acc, label="train"); axes[row, 1].plot(h.epoch, h.val_acc, label="validation")
    axes[row, 0].set_title(f"{key.upper()}: loss"); axes[row, 1].set_title(f"{key.upper()}: accuracy")
    for ax in axes[row]:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
""")

step("Evaluating All Word Models", "The saved predictions of the four models on the clips of the 4 test people.", """
!python evaluate_all.py --task words
""")

step("Word Results", "Accuracy with a 95% bootstrap interval on all test clips, and the accuracy for each of the 4 test people.", """
section("figures/words_metrics.md", "Main results (test people)")
section("figures/words_metrics.md", "Accuracy for each test person")
show("figures/words_model_comparison.png", 800)
""")

step("Statistical Significance (McNemar Test)", "The same test as for the letters, now on the 1,753 test clips.", """
section("figures/words_metrics.md", "McNemar test (exact, two-sided)")
""")

step("F1-score for Each Word and Common Mistakes", "The F1-score of every word and the five most common mistakes of each model.", """
section("figures/words_metrics.md", "F1 for each word")
section("figures/words_metrics.md", "Most common mistakes (true word as predicted, count)")
""")

step("Confusion Matrix", "Confusion matrix of the GRU on the test people.", """
show("figures/words_confusion_gru.png", 750)
""")

step("Testing the Word Model on Example Clips", """
The model of the webcam application (`words_model.pt`, the best network on validation) gives its
3 best guesses for some clips of the test people.
""", """
from word_live import WordModel
word_model = WordModel()
examples = clips[clips.participant_id.isin(test_p)].sample(8, random_state=1)
rows = []
for sid, sign in zip(examples.sequence_id, examples.sign):
    top = word_model.top(np.load(f"data/islr/clips/{sid}.npy").astype(np.float32), aspect=1.2)
    rows.append({"true word": sign, "1st guess": f"{top[0][0]} ({top[0][1]:.0%})",
                 "2nd guess": f"{top[1][0]} ({top[1][1]:.0%})", "3rd guess": f"{top[2][0]} ({top[2][1]:.0%})"})
pd.DataFrame(rows)
""")

md("""
# Webcam Application
`webcam_demo.py` runs the models live on the webcam (`python webcam_demo.py`).

In letters mode the user shows a letter with one or both hands and holds it for about 1.5 s.
The SVM model recognizes it, and the letter is written on the screen and spoken.

In words mode (key M) the user makes a word and then lowers the hands or holds them still.
The word network then shows its 3 best guesses.

We also wrote `word_examples.py`. It plays a real clip of every word from the dataset as a
skeleton, so a user can see how the sign is made before trying it.
""")

step("Model Comparison", "Test accuracy of all models, computed from the saved prediction files.", """
from sklearn.metrics import accuracy_score, f1_score

names = {"svm": "SVM", "rf": "Random Forest", "vit": "ViT (ready model)", "mobilenet_hand": "MobileNetV2, hand crop",
         "mobilenet": "MobileNetV2, whole image", "cnn": "1D-CNN", "gru": "GRU"}
rows = []
for task, folder, keys in [("letters (720 images)", "predictions", ["svm", "rf", "vit", "mobilenet_hand", "mobilenet"]),
                           ("words (1,753 clips)", "predictions_words", ["svm", "rf", "cnn", "gru"])]:
    for k in keys:
        p = pd.read_csv(f"{folder}/{k}.csv", keep_default_na=False).astype({"true": str, "pred": str})
        rows.append({"task": task, "model": names[k], "accuracy": accuracy_score(p.true, p.pred),
                     "macro F1": f1_score(p.true, p.pred, average="macro", labels=sorted(p.true.unique()))})
comparison = pd.DataFrame(rows)
display(comparison.round(3))
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
for ax, (task, g) in zip(axes, comparison.groupby("task", sort=False)):
    ax.barh(g.model, g.accuracy); ax.set_xlim(0, 1); ax.set_title(task); ax.invert_yaxis()
    for y, v in enumerate(g.accuracy):
        ax.text(v + 0.01, y, f"{v:.1%}", va="center")
plt.tight_layout(); plt.show()
""")

md("""
# Discussion
All models were tested on people they never saw in training, and this changed the results a lot.
On validation (the same dataset as training) every letter model got between 96% and 100%. On the
external test set the accuracy dropped to between 59% and 83%.

For the letters, the SVM on the 63 hand landmarks was the best model. It recognized 83.3% of all
720 test images (87.8% when MediaPipe found the hand), and its macro F1 was 0.848. The McNemar test
showed that it is better than the ready ViT (p = 0.04 on all images, p < 0.0001 on the images with
a hand). It is also better than Random Forest and both MobileNetV2 models (p < 0.0001).

MobileNetV2 had the largest gap between validation and test: 99.9% on validation but 71.7% on the
test. The training photos have the same background and light, so a network that sees pixels also
learns the background. Cutting out the hand raised its test accuracy from 59.0% to 71.7%. The
landmark models do not see the background at all. We think this is the main reason why a simple
SVM worked better here than a deep network. The learning curve agrees with this: after about
1,000 images per letter, more images of the same kind did not help on new people.

The most common mistake was N predicted as M, for four of the five models. These two signs differ
only in how many fingers cover the thumb, and in the landmarks this difference is very small.

For the words, the order of the frames was important. The 1D-CNN (83.3%) and the GRU (85.9%) were
much better than SVM (75.4%) and Random Forest (73.2%), which only see a summary of the clip.
All these differences were significant. The accuracy depended a lot on the person. The GRU got
95.4% on one test person and 68.9% on another, because people make the same sign in slightly
different ways. Most mistakes were between signs with a similar movement near the same part of
the body, for example finish and bye, or sad, mad and sleep. The webcam application uses the
1D-CNN, because it was the best network on validation. We did not choose the model by the test set.

The project has some limitations. The letter training data is very uniform, and the letter test
set is small (720 images), so the confidence intervals are about 5 percentage points wide. We used
only 25 words. The webcam application was only checked by hand, not with a measured test.
""")

md("""
# Conclusion
Hand landmarks from MediaPipe worked well as input for sign recognition. A simple SVM on 63
numbers recognized the static letters of new people better (83.3%) than a fine-tuned MobileNetV2
(71.7%) and a ready Vision Transformer (79.6%).

For words with movement, a model has to see the sequence of frames. The GRU recognized 25 words
of 4 new people with 85.9% accuracy, while SVM on a summary of the clip reached 75.4%.

We also learned that validation on the same dataset is not enough. Only the test on new people
showed how much the image models depend on the background.

In the future we would like to add more words, use more varied training data for the letters,
and measure the accuracy of the webcam application on recordings of new users.
""")

if "--markdown" in sys.argv:
    # put the new markdown into the executed notebook, keep the code cells and their output
    old = nbf.read("SignVision.ipynb", 4)
    assert [c.cell_type for c in old.cells] == [c.cell_type for c in cells], "cells differ, run the notebook again"
    for o, n in zip(old.cells, cells):
        if o.cell_type == "markdown":
            o.source = n.source
        else:
            assert o.source == n.source, "code changed, run the notebook again"
    nbf.write(old, "SignVision.ipynb")
    print("Updated the markdown cells of SignVision.ipynb")
    sys.exit()

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "SignVision.ipynb")
print(f"Wrote SignVision.ipynb with {len(cells)} cells")
