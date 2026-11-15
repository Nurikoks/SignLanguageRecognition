# SignVision results

Two tasks: 24 static ASL letters (A to Y without J and Z) from images, and 25 ASL words with
movement from video landmarks. All tables here are copied from the files that
`evaluate_all.py` makes from the saved predictions (`figures/metrics.md` and
`figures/words_metrics.md`). The predictions of every model are in `predictions/` and
`predictions_words/`, so anyone can compute the numbers again with
`python evaluate_all.py` and `python evaluate_all.py --task words` (no data or models needed).

# Part 1. Letters

## Data

| Part | Dataset | Size |
|---|---|---|
| Training and validation | ASL Alphabet (Kaggle, grassknoted/asl-alphabet) | 3,000 images per letter, 72,000 in total |
| External test | ASL Alphabet Test (Kaggle, danrasband/asl-alphabet-test) | 30 images per letter, 720 in total |

The training images are frames of videos, so neighbouring images look almost the same.
We split them in blocks of 50 frames in a row: every fifth block goes to validation, the rest
to training (`results.py`). So the same picture is never in both parts. All models use the same
split.

The external test set has other people, other hands and other backgrounds. It is only used
for the final numbers, never to choose a model or a setting. MediaPipe finds a hand in 683 of
the 720 test images. SVM, Random Forest and MobileNetV2 (hand crop) need the hand, so they give
no answer for the other 37 images. We count these 37 images as mistakes in the "all 720" columns.

## Models

| Model | Input | Training data | How it was trained |
|---|---|---|---|
| SVM | 63 landmark features | 44,772 images with a hand (training part), 10,596 for validation | RBF kernel, C = 10, gamma = scale, StandardScaler |
| Random Forest | 63 landmark features | the same as SVM | 300 trees |
| ViT | whole image, 224x224 | none, we did not train it | ready model dima806/asl_alphabet_image_detection (ViT-Base, 29 classes), we only use the 24 letter outputs |
| MobileNetV2, hand crop | square around the hand, 224x224 | 44,772 for training, 10,596 for validation | ImageNet weights, 2 epochs only the last layer (lr 0.001), then 8 epochs the whole network (lr 0.0003), AdamW, cosine schedule, label smoothing 0.1, mixed precision |
| MobileNetV2, whole image | whole image, 224x224 | 57,600 for training, 14,400 for validation | the same as above |

The 63 features are the 21 MediaPipe hand landmarks (x, y, z), moved to the wrist, scaled by
the hand size and mirrored for the left hand. MobileNetV2 is not a ready model: we start from
ImageNet weights and train it on our letters. Only ViT is used as it is.

## Validation results (same dataset as training)

| Model | Validation accuracy | Validation macro F1 |
|---|---|---|
| SVM | 98.0% | 0.978 |
| Random Forest | 96.5% | 0.961 |
| MobileNetV2, hand crop (best epoch 10 of 10) | 99.9% | 0.999 |
| MobileNetV2, whole image (best epoch 5 of 10) | 100.0% | 1.000 |

ViT has no number here because we did not train it. SVM is better than Random Forest on
validation, so the webcam demo uses SVM (`model.joblib`).

## Main results (external test)

| Model | Accuracy, all 720 | 95% CI (bootstrap) | Accuracy, hand found (683) | Macro precision | Macro recall | Macro F1, all 720 | Macro F1, hand found |
|---|---|---|---|---|---|---|---|
| SVM | **83.3%** (600) | 80.6% to 85.8% | **87.8%** | 0.890 | 0.833 | **0.848** | 0.871 |
| ViT (ready model) | 79.6% (573) | 76.7% to 82.5% | 79.5% | 0.837 | 0.796 | 0.782 | 0.782 |
| Random Forest | 77.4% (557) | 74.3% to 80.4% | 81.6% | 0.847 | 0.774 | 0.784 | 0.807 |
| MobileNetV2, hand crop | 71.7% (516) | 68.3% to 74.9% | 75.5% | 0.778 | 0.717 | 0.709 | 0.726 |
| MobileNetV2, whole image | 59.0% (425) | 55.4% to 62.5% | 58.7% | 0.682 | 0.590 | 0.550 | 0.545 |

The number in brackets is the count of correct answers out of 720. The 95% interval comes
from 2,000 bootstrap samples of the test images.

All models are much worse on the external test than on validation (96 to 100% down to 59 to
83%). The gap is biggest for MobileNetV2: 99.9% on validation but 71.7% on the test. The
training photos have the same person and the same background, so the network also learns the
background. Cutting out the hand helps a lot (59.0% to 71.7%). SVM and Random Forest only see
the landmark coordinates, not the pixels, so the background does not change their input.
ViT is the only model that answers on all 720 images, because it does not need MediaPipe.

Figures: `figures/model_comparison.png` (accuracy with the 95% intervals) and
`figures/confusion_<model>.png` (confusion matrix of each model on the external test).

## Learning curves

`figures/learning_curve_landmarks.png`: SVM and Random Forest trained on 50 to 3,000 images
per letter (`figures/learning_curve_landmarks.csv`).

| Images per letter | SVM validation | SVM external test (hand found) | RF validation | RF external test (hand found) |
|---|---|---|---|---|
| 50 | 94.6% | 82.3% | 91.9% | 74.1% |
| 250 | 96.9% | 85.8% | 95.0% | 80.1% |
| 500 | 97.4% | 87.3% | 95.8% | 80.8% |
| 1000 | 97.8% | 87.8% | 96.4% | 81.8% |
| all (about 1,900) | 98.0% | 87.8% | 96.5% | 81.6% |

On the external test both models stop getting better after about 1,000 images per letter.
More images of the same people do not help with new people.

`figures/learning_curve_mobilenet_hand.png` and `figures/learning_curve_mobilenet.png`: loss
and accuracy of MobileNetV2 for each epoch (`figures/history_*.csv`). Validation accuracy is
above 99% from epoch 3, so the validation set cannot show the overfitting that the external
test shows.

## Are the differences real? (McNemar test)

Exact two-sided McNemar test on the same images. "Only A right" counts the images where model A
is right and model B is wrong. The full table with all pairs is in `figures/metrics.md`.

| Model A | Model B | Images | Only A right | Only B right | p-value |
|---|---|---|---|---|---|
| SVM | Random Forest | all 720 | 62 | 19 | < 0.0001 |
| SVM | ViT | all 720 | 94 | 67 | 0.04 |
| SVM | ViT | hand found | 94 | 37 | < 0.0001 |
| SVM | MobileNetV2, hand crop | all 720 | 115 | 31 | < 0.0001 |
| Random Forest | ViT | all 720 | 80 | 96 | 0.26 |
| Random Forest | MobileNetV2, hand crop | all 720 | 94 | 53 | 0.0009 |
| ViT | MobileNetV2, hand crop | all 720 | 102 | 45 | < 0.0001 |
| MobileNetV2, hand crop | MobileNetV2, whole image | all 720 | 143 | 52 | < 0.0001 |

SVM is significantly better than every other model. Random Forest and ViT are about the same
(p = 0.26). The hand crop really helps MobileNetV2.

## F1 for each letter (external test, all 720 images)

| Letter | SVM | Random Forest | ViT | MobileNetV2, hand crop | MobileNetV2, whole image |
|---|---|---|---|---|---|
| A | 1.00 | 0.97 | 0.88 | 1.00 | 0.17 |
| B | 0.97 | 0.97 | 1.00 | 0.90 | 0.59 |
| C | 0.58 | 0.48 | 0.70 | 0.50 | 0.28 |
| D | 0.86 | 0.65 | 0.97 | 0.75 | 0.51 |
| E | 0.90 | 0.83 | 0.74 | 0.89 | 0.48 |
| F | 1.00 | 1.00 | 0.97 | 0.71 | 0.56 |
| G | 0.78 | 0.91 | 0.90 | 0.97 | 0.71 |
| H | 0.88 | 0.92 | 0.91 | 1.00 | 0.89 |
| I | 0.78 | 0.78 | 0.79 | 0.51 | 0.66 |
| K | 0.84 | 0.83 | 0.70 | 0.55 | 0.57 |
| L | 0.94 | 0.91 | 0.81 | 0.88 | 0.91 |
| M | 0.62 | 0.55 | 0.75 | 0.65 | 0.58 |
| N | 0.43 | 0.39 | 0.49 | 0.27 | 0.46 |
| O | 0.67 | 0.43 | 0.94 | 0.75 | 0.24 |
| P | 0.95 | 0.97 | 0.87 | 0.92 | 0.94 |
| Q | 0.93 | 0.95 | 0.82 | 0.91 | 0.93 |
| R | 0.89 | 0.67 | 0.66 | 0.67 | 0.60 |
| S | 0.78 | 0.91 | 0.67 | 0.42 | 0.12 |
| T | 0.87 | 0.57 | 0.45 | 0.64 | 0.45 |
| U | 0.94 | 0.56 | 0.85 | 0.85 | 0.52 |
| V | 0.85 | 0.75 | 0.50 | 0.00 | 0.12 |
| W | 0.98 | 0.92 | 0.94 | 0.81 | 0.58 |
| X | 0.92 | 0.91 | 0.63 | 0.54 | 0.49 |
| Y | 1.00 | 1.00 | 0.86 | 0.95 | 0.83 |

## Most common mistakes (true letter as predicted, count of 30)

| Model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| SVM | N as M (18) | C as O (12) | I as no answer (10) | S as E (7) | O as no answer (5) |
| Random Forest | N as M (20) | O as D (14) | R as U (12) | C as D (12) | T as E (11) |
| ViT | N as M (18) | V as K (16) | T as X (12) | Q as P (9) | R as K (8) |
| MobileNetV2, hand crop | N as M (23) | V as K (23) | S as I (16) | X as R (12) | I as no answer (10) |
| MobileNetV2, whole image | V as K (22) | O as D (21) | A as W (14) | S as I (13) | R as K (9) |

N as M is the top mistake for four of the five models: these signs differ only in how many
fingers cover the thumb. ViT is the best model on C, D, M and O, where the outline of the whole
hand matters more than the exact finger positions.

# Part 2. Words with movement

## Data

Kaggle competition "Google - Isolated Sign Language Recognition" (asl-signs): 94,477 videos of
250 ASL signs by 21 people, given as MediaPipe Holistic landmarks for each frame. We took 25
everyday words (`words.py`): hello, bye, thank you, please, yes, no, drink, water, food, hungry,
sleep, happy, sad, mad, mom, dad, home, go, wait, look, finish, like, where, why, tomorrow.
That is 9,596 clips (346 to 414 per word). For 326 clips no hand or no shoulders were found,
so 9,270 clips are used.

The clips are split by person, so the test people are never seen in training:

| Part | People | Clips |
|---|---|---|
| Training | 14 | 6,207 |
| Validation | 3 | 1,310 |
| Test | 4 (30680, 37779, 49445, 62590) | 1,753 |

## Features and models

From each frame we keep both hands (21 points) and 7 upper body points, move the middle of the
shoulders to (0, 0) and scale by the shoulder width, mirror left-handed signers, and resample the
clip to 32 steps (`word_features.py`). Every step has 104 numbers: for each hand its wrist
position, the hand shape and a flag "hand seen", plus the body points. Training clips are
augmented 8 times (stretch, rotation, scale, shift, cut in time).

| Model | Input | Settings |
|---|---|---|
| SVM | 1,040 numbers per clip (the sequence averaged in 8 time parts, plus mean and std) | RBF kernel, C = 10, trained on the clips and 2 augmented copies |
| Random Forest | the same | 500 trees |
| 1D-CNN | 32 x 104 sequence | 4 convolution blocks (kernel 5, 192 channels, GELU, dropout 0.3), mean and max pooling over time |
| GRU | 32 x 104 sequence | linear layer to 128, then a 2-layer bidirectional GRU (128), dropout 0.3 |

The networks are trained for 80 epochs (AdamW, one-cycle learning rate, label smoothing 0.1),
and we keep the epoch with the best validation accuracy.

## Results on the 4 test people

| Model | Validation accuracy | Test accuracy (1,753 clips) | 95% CI (bootstrap) | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|---|
| SVM | 74.4% | 75.4% (1321) | 73.4% to 77.4% | 0.764 | 0.747 | 0.747 |
| Random Forest | 69.8% | 73.2% (1283) | 71.1% to 75.2% | 0.735 | 0.725 | 0.722 |
| 1D-CNN | **87.9%** | 83.3% (1460) | 81.5% to 85.0% | 0.842 | 0.829 | 0.831 |
| GRU | 86.6% | **85.9%** (1505) | 84.1% to 87.4% | 0.862 | 0.854 | 0.855 |

The webcam demo uses the 1D-CNN (`words_model.pt`), because it is the best on validation.
GRU is a bit better on the test people, but choosing the model by the test set would make
the test numbers too optimistic, so we keep the choice made on validation.

Figures: `figures/words_model_comparison.png`, `figures/words_confusion_<model>.png`,
`figures/learning_curve_words_cnn.png` and `figures/learning_curve_words_gru.png`.

## Accuracy for each test person

| Model | person 30680 (350 clips) | person 37779 (456 clips) | person 49445 (491 clips) | person 62590 (456 clips) |
|---|---|---|---|---|
| SVM | 59.7% | 92.3% | 59.3% | 87.7% |
| Random Forest | 62.0% | 84.6% | 56.8% | 87.9% |
| 1D-CNN | 72.9% | 88.8% | 77.0% | 92.5% |
| GRU | 68.9% | 93.6% | 81.9% | 95.4% |

The accuracy depends a lot on the person (69% to 95% for GRU): people make the same sign in
different ways. GRU is the best model for every test person except 30680, where the 1D-CNN is better.

## McNemar test (exact, two-sided)

| Model A | Model B | Only A right | Only B right | p-value |
|---|---|---|---|---|
| SVM | Random Forest | 173 | 135 | 0.03 |
| SVM | 1D-CNN | 93 | 232 | < 0.0001 |
| SVM | GRU | 52 | 236 | < 0.0001 |
| Random Forest | 1D-CNN | 75 | 252 | < 0.0001 |
| Random Forest | GRU | 49 | 271 | < 0.0001 |
| 1D-CNN | GRU | 78 | 123 | 0.0018 |

All differences are significant. Models that see the order of the frames (1D-CNN, GRU) are
much better than the summary vector for SVM and Random Forest.

## Hardest words

The lowest F1 for GRU: go (0.63), mad (0.76), bye (0.77), sleep (0.78), like (0.80).
The most common mistakes are finish as bye, like as hungry, and sad or mad as sleep: these
pairs have a similar movement near the same part of the body. F1 for every word and model is
in `figures/words_metrics.md`.

# Notes

- The downloaded letter test folder has a second copy of itself inside. Our scripts take only
  the first folder for each letter, so every test image is counted once (720, not 1,440).
- Results of an earlier run (SVM and Random Forest on 500 images per letter: SVM 81.9%,
  MobileNetV2 whole image 62.9%) are replaced by this run on all images. MobileNetV2 on whole
  images got 59.0% this time with the same settings. This shows that its result also depends on
  the random start of training.
