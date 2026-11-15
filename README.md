# SignVision

Recognizes American Sign Language from a camera in two ways:

- **Letters:** 24 static ASL letters (A to Y without J and Z) from images. We compare SVM and
  Random Forest on 21 MediaPipe hand landmarks, MobileNetV2 fine-tuned on the hand part of the
  image (and on the whole image as an experiment), and a ready ViT model from Hugging Face.
- **Words:** 25 ASL words with movement (hello, thank you, water, ...) from the Kaggle ISLR
  landmark videos. We compare SVM and Random Forest on a summary of the clip with a 1D-CNN and
  a GRU on the whole sequence. The test people are never seen in training.

A webcam demo shows both, works with one or two hands and has a 6-7 easter egg.
All results are in [RESULTS.md](RESULTS.md).

| Task | Best model | Test accuracy |
|---|---|---|
| Letters (720 images of new people) | SVM on landmarks | 83.3% (87.8% when a hand is found) |
| Words (1,753 clips of 4 new people) | GRU | 85.9% |

## Setup
1. `pip install -r requirements.txt`
2. For the neural networks you need PyTorch with CUDA (the default pip version only uses the CPU).
   Pick the CUDA version that your driver supports on https://pytorch.org, for example:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
   ```
3. Download the MediaPipe models into this folder:
   - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   - https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
     (only for words in the webcam demo)
4. Download the data from Kaggle into `data/`:
   - ASL Alphabet (grassknoted/asl-alphabet) to `data/asl_alphabet_train`, for training letters
   - ASL Alphabet Test (danrasband/asl-alphabet-test) to `data/asl-alphabet-test`, only for testing
   - for words: accept the rules of the competition "asl-signs" on kaggle.com, then
     `kaggle competitions download -c asl-signs -p data/islr/full` (about 40 GB, the archive can
     be deleted after `islr_prepare.py`)

The data and the trained models are not in git (too big). The scripts below make them again.
The predictions of all models are in git (`predictions/`, `predictions_words/`), so all tables
and figures can be made again without the data: `python evaluate_all.py`.

## Run

Letters, SVM and Random Forest:
```
python extract_features.py --data data/asl_alphabet_train --out train.csv
python extract_features.py --data data/asl-alphabet-test --out test.csv
python train.py
```
`train.py` also makes the learning curve (models trained on 50 to 3,000 images per letter).
`--no-curve` skips it. The better model on validation is saved as `model.joblib` for the demo.

Letters, neural networks:
```
python train_mobilenet.py --crop-hand --out mobilenet_hand.pt
python train_mobilenet.py --out mobilenet.pt
python evaluate_vit.py
```
The first command trains MobileNetV2 on a square around the hand. MediaPipe finds the hands
once and saves the boxes to `hand_boxes.csv` (about 40 minutes for all images). Training takes
about 45 minutes on an RTX 4060 Laptop GPU. The second command trains on whole images, we kept it
as an experiment. `--eval-only` tests a saved model without training. `evaluate_vit.py` does not
train anything: it downloads the ready model (about 340 MB) and tests it.

Words (about 10 minutes after the download):
```
python islr_prepare.py
python train_islr.py
```
`islr_prepare.py` reads only the clips of our 25 words from the archive and keeps the hands and
the pose. `train_islr.py` trains all 4 word models and saves the best network on validation as
`words_model.pt` for the demo.

All numbers and figures:
```
python evaluate_all.py
python evaluate_all.py --task words
```
They go to `figures/metrics.md`, `figures/words_metrics.md` and `figures/*.png` (confusion
matrices, model comparison). The learning curves are made by the training scripts.

How the data is split:
- Letters: the training images are frames of videos, so they are split in blocks of 50 frames
  (every fifth block is validation). The external test set has other people and backgrounds and
  is only used for the final numbers.
- Words: split by person, 14 people for training, 3 for validation, 4 for the test.

## Demo
```
python webcam_demo.py
```
- Letters mode: show a letter with one or both hands and hold it for about 1.5 s. The letter is
  written and spoken.
- Words mode (press M): make one of the 25 words, then lower your hands or hold them still.
  The 3 best guesses are shown for 3 seconds, the best one is written if the model is sure enough.
  To see how the words are made, run `python word_examples.py`: it plays a real clip of each
  word from the dataset as a skeleton (N / P next / previous word, S slow; `--save` makes mp4 files).
- 6-7: put a photo named `friend.jpg` in this folder (or use `--friend path/to/photo.jpg`).
  Hold both hands palms up and move them up and down in turns (one up, the other down): the photo
  appears for 3 seconds. Works in both modes.

Keys: M switch letters / words, SPACE add a space, BACKSPACE delete, C clear text, Q or Esc quit.

## Camera
By default the camera is asked for 640x480 at 60 fps and gives the most it can.
The real fps is shown in the bottom right corner of the window. The window can be resized.
```
python webcam_demo.py --width 1280 --height 720 --fps 60
```
Many webcams give HD only at a low fps (10 to 15) and 60 fps not at all, so check the counter.
MediaPipe makes the image smaller inside anyway, so a high fps helps more than a big image.
In a dark room auto exposure makes the frames longer: fps drops (e.g. 30 to 15) and moving
hands get blurry. Turn on the light, or set a short manual exposure: `--exposure -6`.

## Files
- `features.py` MediaPipe hand landmarks and their normalization (shared by all scripts)
- `extract_features.py` images to `train.csv` / `test.csv` (63 features, label, file)
- `results.py` the train / validation split of the letter images and the list of test images
- `train.py` SVM and Random Forest on letters, learning curve
- `train_mobilenet.py` fine-tunes MobileNetV2 (ImageNet weights) on whole images or hand crops,
  loss and accuracy per epoch
- `evaluate_vit.py` tests the ready ViT model
- `words.py` the 25 words
- `islr_prepare.py` takes the clips of the words out of the Kaggle ISLR archive
- `word_features.py` features of a word clip (normalization, mirroring, resampling)
- `train_islr.py` SVM, Random Forest, 1D-CNN and GRU on words
- `word_live.py` word recording and the word model for the webcam
- `evaluate_all.py` all metrics and figures from the saved predictions (accuracy with
  bootstrap intervals, macro precision / recall / F1, McNemar test, F1 per class, confusion matrices)
- `webcam_demo.py` live demo
- `word_examples.py` shows a real dataset clip of each word as a skeleton
- `SignVision.ipynb` the notebook with all experiments and their output, built by `make_notebook.py`
  (`--clean` removes MediaPipe log lines, `--pdf` makes `SignVision.pdf`)
- `make_report.py` builds the APA report `Report.docx` and `Report.pdf` (needs Microsoft Word for the PDF)
- `camera.py` opens the webcam with the wanted size and fps, fps counter
- `gestures.py` frame buffer and the 6-7 detector

Note: the downloaded test folder has a second copy of itself inside
(`data/asl-alphabet-test/asl-alphabet-test`). The scripts take only the first folder for each
letter, so every image is counted once.
