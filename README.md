# SignVision

Recognizes 24 static ASL letters (A to Y without J and Z) from 21 MediaPipe hand landmarks
using SVM and Random Forest. Works with one or two hands at the same time, can learn your
own word gestures with movement, and has a 6-7 easter egg.

## Setup
1. `pip install -r requirements.txt`
2. Download the MediaPipe hand model into this folder:
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
3. Download from Kaggle into `data/`:
   - ASL Alphabet (grassknoted/asl-alphabet), for training
   - ASL Alphabet Test (danrasband/asl-alphabet-test), only for testing

## Run
```
python extract_features.py --data data/asl_alphabet_train --out train.csv --per-class 500
python extract_features.py --data data/asl-alphabet-test --out test.csv
python train.py --train train.csv --test test.csv
python webcam_demo.py
```
`--per-class 500` makes extraction fast for a first try. Remove it to use all 3,000 images per letter.

## Words (gestures with movement)
There is no word dataset, so you record your own examples (about 20 per word):
```
python record_words.py HELLO --count 20
python record_words.py THANKS --count 20
python record_words.py nothing --count 20
python train_words.py
python webcam_demo.py
```
In the recorder press SPACE, wait for the countdown and make the gesture while the frame is red
(2 s). `nothing` is optional: random moves that are not a word, so the demo makes fewer mistakes.
Use Latin letters for word names (OpenCV cannot draw other letters).
In the demo press M to switch between letters and words.

## 6-7
Put a photo named `friend.jpg` in this folder (or use `--friend path/to/photo.jpg`).
Hold both hands palms up and move them up and down in turns (one up, the other down):
the photo appears for 3 seconds. Works in both modes, no training needed.

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
`record_words.py` takes the same options.

## Demo keys
M switch letters / words, SPACE add a space, BACKSPACE delete, C clear text, Q or Esc quit.

## Files
- `features.py` MediaPipe landmark extraction and normalization (shared by all scripts)
- `extract_features.py` images to `train.csv` / `test.csv` (63 features + label)
- `train.py` trains SVM and Random Forest, prints metrics, saves `model.joblib` and a confusion matrix
- `webcam_demo.py` live demo: hold a sign for about 1.5 s (each hand on its own), the letter is
  shown and spoken; words mode; 6-7
- `camera.py` opens the webcam with the wanted size and fps, fps counter
- `gestures.py` features for movement gestures with two hands, and the 6-7 detector
- `record_words.py` records word examples with the webcam to `data/words/<WORD>/`
- `train_words.py` trains SVM and Random Forest on the words (cross-validation), saves `words_model.joblib`
