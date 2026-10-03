# SignVision

Recognizes 24 static ASL letters (A to Y without J and Z) from 21 MediaPipe hand landmarks
using SVM and Random Forest.

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

## Files
- `features.py` MediaPipe landmark extraction and normalization (shared by all scripts)
- `extract_features.py` images to `train.csv` / `test.csv` (63 features + label)
- `train.py` trains SVM and Random Forest, prints metrics, saves `model.joblib` and a confusion matrix
- `webcam_demo.py` live demo: hold a sign for about 1.5 s, the letter is shown and spoken
