# SignVision results

All results of the four models for 24 static ASL letters (A to Y without J and Z). We also
include a second MobileNetV2 trained on whole images, which we kept as an experiment.
We computed all numbers again in one run on 12.11.2026, so every model is tested on
exactly the same images.

## Data

| Part | Dataset | Size |
|---|---|---|
| Training | ASL Alphabet (Kaggle, grassknoted/asl-alphabet) | 3,000 images per letter, 72,000 in total |
| External test | ASL Alphabet Test (Kaggle, danrasband/asl-alphabet-test) | 30 images per letter, 720 in total |

The external test set has other people, other hands and other backgrounds, so it shows how
the models work on new photos. MediaPipe finds a hand in 683 of the 720 test images.
SVM, Random Forest and MobileNetV2 (hand crop) need the hand, so they give no answer for the
other 37 images. We count these 37 images as mistakes in the "all 720" columns.

## Models

| Model | Input | Training data | How it was trained |
|---|---|---|---|
| SVM | 63 landmark features | 500 images per letter, 7,553 with a hand for training, 1,889 for the internal test | RBF kernel, C = 10, gamma = scale, features scaled with StandardScaler |
| Random Forest | 63 landmark features | the same as SVM | 300 trees |
| ViT | whole image, 224x224 | none, we did not train it | ready model dima806/asl_alphabet_image_detection (ViT-Base, 29 classes), we only use the 24 letter outputs |
| MobileNetV2, hand crop | square around the hand, 224x224 | all images with a hand: 44,772 for training, 10,596 for validation | ImageNet weights, 2 epochs only the last layer (lr 0.001), then 8 epochs the whole network (lr 0.0003), AdamW, cosine schedule, label smoothing 0.1 |
| MobileNetV2, whole image | whole image, 224x224 | 57,600 for training, 14,400 for validation | the same as above |

The 63 features are the 21 MediaPipe hand landmarks (x, y, z), normalized for position and size.
The training images are frames of videos, so for MobileNetV2 we split them in blocks of
50 frames in a row. Otherwise almost the same picture would be in training and validation.

## Results on the training data

These numbers come from a part of the training dataset that the model did not see in training.

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM | 98.1% | 0.979 |
| Random Forest | 97.7% | 0.976 |
| MobileNetV2, hand crop (best epoch, 9 of 10) | 99.9% | 0.999 |
| MobileNetV2, whole image (best epoch, 9 of 10) | 100.0% | 1.000 |

ViT has no number here because we did not train it.

## Main results (external test)

| Model | Accuracy, all 720 | Accuracy, hand found (683) | Macro precision, all | Macro recall, all | Macro F1, all 720 | Macro F1, hand found |
|---|---|---|---|---|---|---|
| SVM | 81.9% (590) | 86.4% | 0.873 | 0.819 | 0.827 | 0.849 |
| ViT (ready model) | 79.6% (573) | 79.5% | 0.837 | 0.796 | 0.782 | 0.782 |
| Random Forest | 77.5% (558) | 81.7% | 0.847 | 0.775 | 0.786 | 0.809 |
| MobileNetV2, hand crop | 70.0% (504) | 73.8% | 0.798 | 0.700 | 0.693 | 0.709 |
| MobileNetV2, whole image | 62.9% (453) | 63.0% | 0.725 | 0.629 | 0.600 | 0.601 |

The number in brackets is the count of correct answers out of 720.

SVM is the best model in every column. All models are much worse on the external test than on
the training data (from 98 to 100% down to 63 to 82%). The gap is biggest for MobileNetV2: it
gets 99.9% on validation but 70.0% on the test. The training photos are very similar to each
other (the same person and the same background), so the network also learns the background.
Cutting out the hand improved the test accuracy from 62.9% to 70.0%. SVM and Random Forest
only see the landmark coordinates, not the pixels, so the background does not change their input.

ViT is the only model that works on all 720 images, because it does not need MediaPipe. On the
683 images with a hand it is behind SVM and Random Forest.

## Are the differences real? (McNemar test)

We compared every pair of models on the same images. "Only A right" counts the images where
model A is right and model B is wrong. A p-value below 0.05 means the difference is probably
not random.

| Model A | Model B | Images | Only A right | Only B right | p-value |
|---|---|---|---|---|---|
| SVM | Random Forest | all 720 | 58 | 26 | 0.0006 |
| SVM | ViT | all 720 | 93 | 76 | 0.22 |
| SVM | ViT | hand found | 93 | 46 | 0.0001 |
| SVM | MobileNetV2, hand crop | all 720 | 118 | 32 | < 0.0001 |
| SVM | MobileNetV2, whole image | all 720 | 192 | 55 | < 0.0001 |
| Random Forest | ViT | all 720 | 78 | 93 | 0.28 |
| Random Forest | ViT | hand found | 78 | 63 | 0.24 |
| Random Forest | MobileNetV2, hand crop | all 720 | 102 | 48 | < 0.0001 |
| ViT | MobileNetV2, hand crop | all 720 | 108 | 39 | < 0.0001 |
| ViT | MobileNetV2, whole image | all 720 | 161 | 41 | < 0.0001 |
| MobileNetV2, hand crop | MobileNetV2, whole image | all 720 | 112 | 61 | 0.0001 |

SVM is better than Random Forest and both MobileNetV2 models. SVM and ViT are not
significantly different on all 720 images (p = 0.22), but on the images with a hand SVM is
clearly better (p = 0.0001). Random Forest and ViT are about the same. The hand crop really
helped MobileNetV2 (p = 0.0001).

## F1 for each letter (external test, all 720 images)

| Letter | SVM | Random Forest | ViT | MobileNetV2, hand crop | MobileNetV2, whole image |
|---|---|---|---|---|---|
| A | 1.00 | 0.97 | 0.88 | 0.89 | 0.39 |
| B | 0.95 | 0.98 | 1.00 | 0.94 | 0.73 |
| C | 0.55 | 0.51 | 0.70 | 0.38 | 0.43 |
| D | 0.78 | 0.76 | 0.97 | 0.69 | 0.60 |
| E | 0.91 | 0.86 | 0.74 | 0.87 | 0.61 |
| F | 0.98 | 0.95 | 0.97 | 0.75 | 0.52 |
| G | 0.89 | 0.95 | 0.90 | 0.97 | 0.82 |
| H | 0.91 | 0.98 | 0.91 | 0.97 | 0.83 |
| I | 0.77 | 0.77 | 0.79 | 0.52 | 0.58 |
| K | 0.83 | 0.87 | 0.70 | 0.63 | 0.53 |
| L | 0.94 | 0.91 | 0.81 | 0.94 | 0.95 |
| M | 0.59 | 0.58 | 0.75 | 0.66 | 0.65 |
| N | 0.12 | 0.33 | 0.49 | 0.32 | 0.59 |
| O | 0.75 | 0.64 | 0.94 | 0.76 | 0.38 |
| P | 0.91 | 0.95 | 0.87 | 0.89 | 0.92 |
| Q | 0.90 | 0.95 | 0.82 | 0.85 | 0.93 |
| R | 0.85 | 0.56 | 0.66 | 0.65 | 0.75 |
| S | 0.81 | 0.90 | 0.67 | 0.42 | 0.12 |
| T | 0.91 | 0.64 | 0.45 | 0.55 | 0.49 |
| U | 0.84 | 0.39 | 0.85 | 0.82 | 0.65 |
| V | 0.75 | 0.64 | 0.50 | 0.12 | 0.06 |
| W | 0.98 | 0.90 | 0.94 | 0.78 | 0.62 |
| X | 0.95 | 0.91 | 0.63 | 0.38 | 0.56 |
| Y | 1.00 | 0.98 | 0.86 | 0.88 | 0.69 |

Letters that are hard for almost all models: N, C and M. N is very low for SVM (0.12) because
SVM almost always calls it M. These two signs differ only in how many fingers cover the thumb,
and the landmarks for this look very similar. ViT is the best model on C, D, M and O, probably
because there the outline of the whole hand matters more than the exact finger positions.

## Most common mistakes

Each line is true letter, then the predicted letter, then how many of the 30 test images.

| Model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| SVM | N as M (25) | C as D (10) | V as K (7) | U as R (7) | S as E (6) |
| Random Forest | N as M (21) | R as U (13) | U as R (11) | T as E (9) | C as O (8) |
| ViT | N as M (18) | V as K (16) | T as X (12) | Q as P (9) | R as K (8) |
| MobileNetV2, hand crop | N as M (21) | X as R (18) | S as I (17) | V as K (16) | C as D (13) |
| MobileNetV2, whole image | V as K (20) | S as I (19) | O as D (17) | T as Y (13) | X as I (9) |

N as M is the top mistake for four of the five models.

## Notes

- The SVM and Random Forest numbers use 500 training images per letter. MobileNetV2 uses all 3,000.
- The downloaded test folder has a second copy of itself inside. Our scripts take only the first
  folder for each letter, so every test image is counted once (720, not 1,440).
- The confusion matrices are saved as `confusion_matrix_svm.png`, `confusion_matrix_vit.png`,
  `confusion_matrix_mobilenet_hand.png` and `confusion_matrix_mobilenet.png` after running the
  scripts. They are not in git.
- In this run the McNemar p-value for SVM against ViT on all 720 images is 0.22. The README says
  0.25 because it comes from an earlier ViT run where one image was different.
