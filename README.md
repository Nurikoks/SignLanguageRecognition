# SignVision

Recognizes 24 static ASL letters (A to Y without J and Z) from images. We compare four models:
SVM and Random Forest on 21 MediaPipe hand landmarks, MobileNetV2 fine-tuned on the hand part
of the image, and a ready ViT model from Hugging Face. There is also a webcam demo that works
with one or two hands at the same time, can learn your own word gestures with movement,
and has a 6-7 easter egg.

## Setup
1. `pip install -r requirements.txt`
2. For MobileNetV2 training you need PyTorch with CUDA (the default pip version only uses the CPU).
   Pick the CUDA version that your driver supports on https://pytorch.org, for example:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
   ```
3. Download the MediaPipe hand model into this folder:
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
4. Download from Kaggle into `data/`:
   - ASL Alphabet (grassknoted/asl-alphabet), for training
   - ASL Alphabet Test (danrasband/asl-alphabet-test), only for testing

## Run
SVM and Random Forest:
```
python extract_features.py --data data/asl_alphabet_train --out train.csv --per-class 500
python extract_features.py --data data/asl-alphabet-test --out test.csv
python train.py --train train.csv --test test.csv
python webcam_demo.py
```
`--per-class 500` makes extraction fast for a first try. Remove it to use all 3,000 images per letter.

Neural networks:
```
python train_mobilenet.py --crop-hand --out mobilenet_hand.pt
python train_mobilenet.py --out mobilenet.pt
python evaluate_vit.py
```
The first command trains MobileNetV2 on a square around the hand. MediaPipe finds the hands
once and saves the boxes to `hand_boxes.csv` (this takes about 40 minutes for all images).
Training takes about 45 minutes on an RTX 4060 Laptop GPU. The second command trains on
whole images, we kept it as an experiment. `--eval-only` tests a saved model without training.
`evaluate_vit.py` does not train anything. It downloads the ready model (about 340 MB) and tests it.

## Results
All numbers are on the external test set (ASL Alphabet Test, 30 images per letter, 720 images).
This set has different people and backgrounds, so it shows how the models work on new photos.
MediaPipe finds a hand in 683 of the 720 images. SVM, Random Forest and the hand crop MobileNet
cannot classify the other 37, so we count them as mistakes in the first column.

| Model | Input | Accuracy, all 720 | Accuracy, hand found (683) | Macro F1, hand found |
|---|---|---|---|---|
| SVM (RBF) | landmarks | 81.9% | 86.4% | 0.849 |
| Random Forest | landmarks | 77.5% | 81.7% | 0.809 |
| ViT (ready model) | whole image | 79.6% | 79.5% | 0.782 |
| MobileNetV2 | hand crop | 70.0% | 73.8% | 0.709 |
| MobileNetV2 (experiment) | whole image | 62.9% | 63.0% | 0.601 |

SVM and Random Forest were trained on 500 images per letter, MobileNetV2 on all 3,000.
SVM is the best model. The difference between SVM and ViT on all 720 images is not significant
(McNemar test, p = 0.25), but on the 683 images with a hand SVM is clearly better (p = 0.0001).

MobileNetV2 gets 99.9% on its validation part, but only 70% on the external test. The training
photos all look very similar, so the network learns the background and the person, not only the
hand. Cutting out the hand improved the test accuracy from 62.9% to 70.0%. The landmark models
do not see the background at all, which is probably why they work better on new photos.
The hardest letters for most models are N, C, S and V. N is often taken for M.

Note: the downloaded test folder has a second copy of itself inside (`data/asl-alphabet-test/asl-alphabet-test`).
The scripts take only the first folder for each letter, so every image is counted once.

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
- `train_mobilenet.py` fine-tunes MobileNetV2 (ImageNet weights) on whole images or hand crops,
  saves the model (`.pt`) and a confusion matrix
- `evaluate_vit.py` tests the ready ViT model and saves a confusion matrix
- `webcam_demo.py` live demo: hold a sign for about 1.5 s (each hand on its own), the letter is
  shown and spoken; words mode; 6-7
- `camera.py` opens the webcam with the wanted size and fps, fps counter
- `gestures.py` features for movement gestures with two hands, and the 6-7 detector
- `record_words.py` records word examples with the webcam to `data/words/<WORD>/`
- `train_words.py` trains SVM and Random Forest on the words (cross-validation), saves `words_model.joblib`
