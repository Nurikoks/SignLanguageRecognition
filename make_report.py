"""Build the APA report Report.docx (and Report.pdf with Microsoft Word).

Usage:
  python make_report.py            # figures for the report, Report.docx, Report.pdf

All numbers in the text come from figures/metrics.md, figures/words_metrics.md,
figures/learning_curve_landmarks.csv and the training logs (see RESULTS.md).
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

FIG = Path("figures")
OUT = Path("Report.docx")

TITLE = "Sign Language Recognition with Hand Landmarks and Neural Networks"
AUTHORS = "Nurassyl Kaziyev, Alua Muratova"
COURSE = "Machine Learning Algorithms"
INSTRUCTOR = "Assiya Karatay"
YEAR = "2026-2027"


# ---------------------------------------------------------------- extra figures

def figure_examples(path):
    """Training and test images of two letters with the MediaPipe landmarks."""
    from extract_features import find_class_folders
    from features import create_detector, detect, draw_hand
    from results import list_images, test_images

    detector = create_detector()
    train = find_class_folders("data/asl_alphabet_train")
    test = test_images()
    items = [(list_images(train["A"])[100], "training image, A"),
             (list_images(train["N"])[100], "training image, N"),
             (next(p for p, l in test if l == "A"), "test image, A"),
             (next(p for p, l in test if l == "N"), "test image, N")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (p, title) in zip(axes, items):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        points, _ = detect(detector, rgb)
        if points is not None:
            draw_hand(rgb, points, (255, 120, 0))
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_word(path, word="hello"):
    """Four frames of a real ISLR clip of a word, drawn as a skeleton."""
    import pandas as pd
    import word_examples as we
    from train_islr import split_people
    from word_features import ISLR_ASPECT
    from word_live import WordModel
    from words import ISLR_DIR

    model = WordModel()
    meta = pd.read_csv(ISLR_DIR / "clips.csv")
    _, _, train_people = split_people(meta["participant_id"].unique())
    g = meta[(meta["sign"] == word) & meta["participant_id"].isin(train_people)]
    best, best_clip = -1.0, None
    for sid in g["sequence_id"].sample(min(we.CANDIDATES, len(g)), random_state=0):
        clip = np.load(ISLR_DIR / "clips" / f"{sid}.npy").astype(np.float32)
        if len(clip) < we.MIN_FRAMES:
            continue
        seen = np.isfinite(clip[:, [0, 21], 0]).any(axis=1).mean()
        prob = dict(model.top(clip, ISLR_ASPECT, k=len(model.words))).get(word, 0.0)
        if prob * seen > best:
            best, best_clip = prob * seen, clip
    px = we.to_pixels(best_clip)
    idx = np.flatnonzero(np.isfinite(px[:, [0, 21], 0]).any(axis=1))
    frames = idx[np.linspace(0, len(idx) - 1, 4).round().astype(int)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for k, (ax, t) in enumerate(zip(axes, frames)):
        img = we.draw(px[t], word, 0.0, 0, 1, False)[80:-40]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {t + 1} of {len(px)}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- text

ABSTRACT = """This project investigates the recognition of American Sign Language (ASL) from a standard camera. Two tasks were addressed: 24 static letters from images and 25 dynamic words from video. For the letters, a Support Vector Machine (SVM) and a Random Forest were trained on 21 hand landmarks extracted with MediaPipe. They were compared with a fine-tuned MobileNetV2 and a pretrained Vision Transformer (ViT). The letter models were trained on the Kaggle ASL Alphabet dataset and evaluated on a separate set of 720 images with different signers and backgrounds. For the words, landmark sequences from the Kaggle Isolated Sign Language Recognition dataset were used to train an SVM, a Random Forest, a 1D convolutional network (1D-CNN) and a GRU. The word data was split by signer, so the four test signers were not seen during training. The models were evaluated with accuracy, macro precision, macro recall and macro F1-score, bootstrap confidence intervals and the McNemar test. A model was considered successful if it reached an accuracy of at least 85% and a macro F1-score of at least 0.80. The SVM was the best letter model with 83.3% accuracy and a macro F1-score of 0.848. It met both criteria (87.8% and 0.871) on the images where a hand was detected. MobileNetV2 reached 99.9% on validation but only 71.7% on the external test. For the words, the GRU met both criteria with 85.9% accuracy and a macro F1-score of 0.855."""

KEYWORDS = "sign language recognition, MediaPipe, hand landmarks, SVM, MobileNetV2, GRU, McNemar test"

INTRODUCTION = [
    """Sign language is the primary language of many deaf and hard of hearing people. Most hearing people do not know it, which makes everyday communication difficult. A system that recognizes signs from an ordinary webcam could support simple tasks, such as learning the alphabet or understanding short phrases. For this reason, sign language recognition is an active research topic in computer vision and machine learning.""",
    """Automatic recognition of signs is a difficult problem. The same letter can look different depending on the hand, the signer, the background and the lighting. Many signs also contain movement, so a single image is not sufficient and the model has to analyse a sequence of frames. In addition, public datasets are often recorded under very similar conditions. A model can therefore achieve a high score on data from its own dataset and still fail on new signers.""",
    """In this project, a recognition system for American Sign Language (ASL) was developed in two parts. The first part classifies 24 static letters of the ASL alphabet from images. The second part classifies 25 everyday words that contain movement, such as hello, thank you, water and tomorrow. For the letters, classical machine learning on hand landmarks (SVM and Random Forest) was compared with a fine-tuned convolutional network (MobileNetV2) and a pretrained Vision Transformer. For the words, SVM and Random Forest were compared with two neural networks for sequences, a 1D-CNN and a GRU.""",
    """The main objective was to determine which approach generalizes best to signers that the model has not seen. Therefore, all final results were measured on data from new signers: an external test set for the letters and four held-out signers for the words. Before the experiments, two success criteria were defined. A model is considered successful if it reaches an accuracy of at least 85% and a macro F1-score of at least 0.80 on this test data. Statistical tests were used to check whether the differences between models are reliable. Finally, a webcam application was built that uses the selected models to recognize letters and words in real time.""",
]

LITERATURE = [
    """Hand pose estimation is an important first step for many sign recognition systems. Zhang et al. (2020) presented MediaPipe Hands, a real-time pipeline that consists of a palm detector and a hand landmark model. The landmark model predicts 21 points of the hand, each with x and y coordinates and a relative depth. The authors trained the model on about 6,000 real-world images, 10,000 gesture images and 100,000 synthetic images, and the pipeline runs in real time on mobile devices. They also noted that a sequence of landmarks can be used to recognize dynamic gestures. This makes the landmarks a compact input for both static and dynamic sign recognition.""",
    """An alternative is to learn directly from image pixels with a convolutional neural network. Sandler et al. (2018) introduced MobileNetV2, which is based on inverted residual blocks and linear bottlenecks. The standard model has about 3.4 million parameters and reaches 72.0% top-1 accuracy on ImageNet, which makes it suitable for devices with limited computing power. Because of its small size, MobileNetV2 is a common choice for fine-tuning on smaller image datasets.""",
    """For signs with movement, larger video datasets have been collected. Li et al. (2020) introduced WLASL, a word-level ASL dataset with 21,083 videos of 2,000 words performed by 119 signers. The dataset was designed for signer-independent evaluation, so the test signers differ from the training signers. The authors compared models that use the image appearance with models that use body pose keypoints, and they reported that recognizing a large vocabulary from new signers remains difficult. Google later published the Isolated Sign Language Recognition dataset on Kaggle (Chow et al., 2023). In this dataset, each video of an isolated sign is already given as MediaPipe landmarks, which allows sequence models to be trained without processing raw video.""",
    """A known risk in deep learning is shortcut learning. Geirhos et al. (2020) define shortcuts as decision rules that perform well on standard benchmarks but fail under different test conditions. One of their examples is a network that does not recognize a cow on a beach, because it has learned to associate cows with grass. The authors recommend testing models on out-of-distribution data to reveal such shortcuts. In sign recognition, a uniform background in the training images can become this kind of shortcut.""",
    """Finally, a reliable comparison of models requires an appropriate statistical test. Dietterich (1998) compared five approximate tests for deciding whether one learning algorithm outperforms another. Two widely used tests showed a high probability of Type I error, which means that they often detect a difference that does not exist. For algorithms that can be trained only once, McNemar's test was the only test with an acceptable Type I error. Since each of our models was trained once and evaluated on the same test set, the McNemar test was selected for this project.""",
]

DATA = [
    """For the letters, two Kaggle datasets were used (Table 1). The ASL Alphabet dataset contains 29 classes with 3,000 images each: the 26 letters A to Z and three additional classes, SPACE, DELETE and NOTHING. Only 24 of these classes were used. The letters J and Z were excluded because in ASL they are produced with a movement of the hand, so they cannot be represented correctly by a single static image. The classes SPACE, DELETE and NOTHING were excluded because they are not letters of the ASL alphabet. They are additional classes that are useful for real-time applications such as typing, and NOTHING contains images without a hand. As a result, the training set contains 72,000 images instead of 87,000. The ASL Alphabet Test dataset was used for the final evaluation. It contains 30 images per class with different signers, hands and backgrounds, and after the same selection of 24 classes it includes 720 images. Figure 1 shows that the training and test images look very different. The test set was used only for the final evaluation and never for model or parameter selection.""",
    """The training images are frames of videos, so neighbouring images are almost identical. A random split would place almost the same picture in both the training and the validation part, and the validation score would be too optimistic. Therefore, the frames were split into blocks of 50 consecutive frames. Every fifth block was assigned to validation and the remaining blocks to training, which gives an 80/20 split. All letter models used the same split.""",
    """MediaPipe Hands (Zhang et al., 2020) was applied to every image. It detected a hand in 55,368 of the 72,000 training images (76.9%) and in 683 of the 720 test images. The 21 landmarks of the hand were converted into 63 features (the x, y and z coordinates of each point). The wrist was moved to the origin, all coordinates were divided by the size of the hand, and the left hand was mirrored. After this normalization, the features do not depend on the position of the hand in the image, on its size or on which hand is used. Images without a detected hand were excluded from the training of the landmark models. In the test set they are counted as errors, because the model cannot give an answer.""",
    """For the words, the Google Isolated Sign Language Recognition dataset (Chow et al., 2023) was used. It contains 94,477 short videos of 250 isolated ASL signs performed by 21 signers. The videos are provided as MediaPipe Holistic landmarks for every frame: 468 face points, 21 points for each hand and 33 body pose points. Twenty-five everyday words were selected: hello, bye, thank you, please, yes, no, drink, water, food, hungry, sleep, happy, sad, mad, mom, dad, home, go, wait, look, finish, like, where, why and tomorrow. This selection contains 9,596 clips, with 346 to 414 clips per word. The median clip is only 21 frames long. The two hands and the body pose (75 points) were kept, and the face points were removed. In 326 clips no hand or no shoulders were detected, so 9,270 clips were used.""",
    """The word clips were split by signer with a fixed random seed. Fourteen signers were used for training (6,207 clips), three for validation (1,310 clips) and four for testing (1,753 clips). In this way, the test results show how the models perform for signers they have never seen. Figure 2 shows one clip of the word hello.""",
    """All clips were preprocessed in the same way. First, the midpoint of the shoulders was moved to the origin and all points were divided by the shoulder width. As a result, the position of the signer in the frame and the distance to the camera do not influence the features. If the left hand was visible in more frames than the right hand, the clip was mirrored, so that signs of left-handed signers look like signs of right-handed signers. Then the clip was cut to the frames in which a hand is visible and resampled to 32 time steps. Each time step contains 104 values. For each hand, these are the wrist position, the hand shape (21 points relative to the wrist) and a flag that shows whether the hand is visible. The remaining values are seven points of the upper body.""",
    """To increase the amount of training data, random transformations were applied to the training clips. They include a vertical stretch between 0.85 and 1.2, a rotation of up to 0.15 radians, scaling between 0.9 and 1.1 and a small shift. In addition, up to 10% of the frames were removed at the start and at the end of the clip. Eight augmented copies were created for each training clip.""",
]

TABLE_DATA = [["Dataset", "Task", "Use", "Size"],
              ["ASL Alphabet (Kaggle)", "letters", "training and validation", "72,000 images, 24 of 29 classes"],
              ["ASL Alphabet Test (Kaggle)", "letters", "external test", "720 images, 24 of 29 classes"],
              ["Isolated Sign Language Recognition (Kaggle)", "words", "training, validation, test", "9,596 clips, 25 of 250 signs, 21 signers"]]

METHODS = [
    """Five models were trained and evaluated for the letters. The first two models use the 63 landmark features. The SVM uses an RBF kernel with C = 10, and the features are standardized before training. The Random Forest consists of 300 trees. Both models were trained with Scikit-learn on all 44,772 training images with a detected hand. To analyse the influence of the training set size, both models were also trained on 50, 100, 250, 500, 1,000, 1,500 and 2,000 images per letter, and a learning curve was produced.""",
    """The third and the fourth model are MobileNetV2 networks (Sandler et al., 2018) that were initialized with ImageNet weights and fine-tuned with PyTorch. One network receives the whole image. The other receives a square region around the hand detected by MediaPipe, which is 1.4 times larger than the hand. The input size is 224 x 224 pixels. First, only the last layer was trained for 2 epochs with a learning rate of 0.001. Then the whole network was trained for 8 epochs with a learning rate of 0.0003. Training used the AdamW optimizer, a cosine learning rate schedule and label smoothing of 0.1. Strong augmentation was applied, including random cropping, flipping, rotation, colour changes and random erasing. The epoch with the best validation macro F1-score was selected. Training one network took about 45 minutes on an NVIDIA GeForce RTX 4060 Laptop GPU.""",
    """The fifth model is a pretrained Vision Transformer from Hugging Face, dima806/asl_alphabet_image_detection. Its author trained it on the same ASL Alphabet dataset with all 29 classes. This model was not retrained in our project. Only its outputs for the 24 selected letters were used. Because the model works on the whole image and does not depend on MediaPipe, it produces an answer for all 720 test images.""",
    """Four models were trained for the words. The SVM (RBF kernel, C = 10) and the Random Forest (500 trees) cannot process a sequence directly, so they receive a summary of the clip. The summary consists of the 32 x 104 sequence averaged over 8 time segments, together with the mean and the standard deviation over the whole clip (1,040 values). These two models were trained on the original clips and two augmented copies of each clip. The 1D-CNN contains four convolution blocks with kernel size 5, 192 channels, batch normalization, GELU activation and dropout of 0.3. It is followed by pooling over time that combines the mean and the maximum. The GRU first projects every time step to 128 values and then applies a two-layer bidirectional GRU with 128 units. Both networks were trained for 80 epochs with AdamW, a one-cycle learning rate schedule with a maximum of 0.002, a batch size of 64 and label smoothing of 0.1. A new random augmentation was applied in every epoch. The epoch with the best validation accuracy was selected, and the better network on validation is used in the webcam application.""",
    """All models were evaluated on the same test data with four metrics: accuracy, macro precision, macro recall and macro F1-score (Géron, 2019). Accuracy shows the share of correctly classified test items, and it is easy to interpret. However, accuracy alone gives more weight to classes with more test items. Macro averaging computes precision, recall and F1-score for each class separately and then takes their mean, so every letter and every word has the same importance in the final score. This property was important in our project because the number of test items per class is not always equal. In the word test set, the number of clips per word differs, and in the letter test set some images of certain letters have no detected hand. With macro averaging, a model cannot obtain a high score by recognizing only the frequent or easy classes. The F1-score combines precision and recall in one value, so it shows both how often a predicted class is correct and how many items of a class are found.""",
    """Based on these metrics, two success criteria were defined before the final evaluation. A model is considered successful if its accuracy on the test data is at least 85% and its macro F1-score is at least 0.80. The accuracy criterion ensures that most signs are recognized correctly. The macro F1 criterion ensures that the performance is acceptable across all classes and not only on average. For the letters, the criteria were checked on all 720 test images and on the 683 images with a detected hand.""",
    """In addition, a 95% confidence interval for the accuracy was computed with 2,000 bootstrap samples of the test set. To test whether one model is better than another, the exact McNemar test was used (Dietterich, 1998). The test only considers the test items where one model is correct and the other model is wrong. Confusion matrices and lists of the most frequent errors were also produced (Géron, 2019). All metrics were computed by one script from the saved predictions of each model.""",
    """Finally, a webcam application was built with OpenCV. In the letter mode, MediaPipe detects one or two hands in each frame, and the SVM recognizes a letter when the hand is held still for about 1.5 seconds. In the word mode, MediaPipe also estimates the body pose. The application records a sign from the moment a hand appears until the hands are lowered or stop moving, and then the word network shows its three most likely words.""",
]

RESULTS_LETTERS = [
    """Table 2 presents the results of the five letter models on the external test set. The SVM on landmarks achieved the best result in every column. It classified 600 of the 720 test images correctly (83.3%) and 87.8% of the images with a detected hand. The pretrained ViT was second with 79.6%, and the Random Forest was third with 77.4%. Both MobileNetV2 models showed the lowest results, with 71.7% for the hand crop and 59.0% for the whole image. Figure 5 shows the same results with the confidence intervals.""",
    """The validation results were very different. The SVM reached 98.0% and the Random Forest 96.5% on validation. MobileNetV2 reached 99.9% with the hand crop and 100.0% with the whole image. All models therefore lost accuracy on new signers, and MobileNetV2 showed the largest drop. Figure 4 shows the training history of MobileNetV2 with the hand crop. Its validation accuracy is above 99% from the third epoch, so the validation set could not reveal the problem.""",
    """Figure 3 shows the learning curves of the landmark models. With only 50 images per letter, the SVM already reached 94.6% on validation and 82.3% on the test images with a detected hand. With 1,000 images per letter, it reached 97.8% on validation and 87.8% on the test. After this point, the validation accuracy increased slightly, but the test accuracy did not change. The Random Forest showed the same pattern at a lower level, with a test accuracy that increased from 74.1% to about 81.6%.""",
]

TABLE_LETTERS = [["Model", "Accuracy (720)", "95% CI", "Accuracy, hand found (683)", "Macro precision", "Macro recall", "Macro F1"],
                 ["SVM (landmarks)", "83.3%", "80.6-85.8%", "87.8%", "0.890", "0.833", "0.848"],
                 ["ViT (pretrained)", "79.6%", "76.7-82.5%", "79.5%", "0.837", "0.796", "0.782"],
                 ["Random Forest (landmarks)", "77.4%", "74.3-80.4%", "81.6%", "0.847", "0.774", "0.784"],
                 ["MobileNetV2, hand crop", "71.7%", "68.3-74.9%", "75.5%", "0.778", "0.717", "0.709"],
                 ["MobileNetV2, whole image", "59.0%", "55.4-62.5%", "58.7%", "0.682", "0.590", "0.550"]]

RESULTS_LETTERS_2 = [
    """Table 3 reports the McNemar test for the most important pairs of models. The SVM was significantly better than the Random Forest and both MobileNetV2 models (p < 0.0001). The difference between the SVM and the ViT was smaller but still significant. The p-value was 0.04 on all 720 images and below 0.0001 on the 683 images with a detected hand. The Random Forest and the ViT were not significantly different (p = 0.26). The hand crop made MobileNetV2 significantly better than the version that uses the whole image (p < 0.0001).""",
]

TABLE_MCNEMAR = [["Model A", "Model B", "Only A correct", "Only B correct", "p-value"],
                 ["SVM", "Random Forest", "62", "19", "< 0.0001"],
                 ["SVM", "ViT", "94", "67", "0.04"],
                 ["SVM", "ViT (hand found)", "94", "37", "< 0.0001"],
                 ["SVM", "MobileNetV2, hand crop", "115", "31", "< 0.0001"],
                 ["Random Forest", "ViT", "80", "96", "0.26"],
                 ["MobileNetV2, hand crop", "MobileNetV2, whole image", "143", "52", "< 0.0001"]]

RESULTS_LETTERS_3 = [
    """Figure 6 shows the confusion matrix of the SVM. Most letters are recognized well, and the F1-score is 1.00 for A, F and Y. The most difficult letter is N, with an F1-score of 0.43. The most frequent error of the SVM is N predicted as M (18 of 30 images). This is also the most frequent error of the Random Forest, the ViT and MobileNetV2 with the hand crop. Other frequent errors of the SVM are C predicted as O (12 images) and S predicted as E (7 images). MediaPipe did not detect a hand in 37 test images, and 15 of them belong to the letters I (10 images) and O (5 images).""",
]

RESULTS_WORDS = [
    """Table 4 presents the results of the four word models on the 1,753 clips of the four test signers. The GRU achieved the best result with 85.9% accuracy and a macro F1-score of 0.855. The 1D-CNN was close with 83.3%. The SVM (75.4%) and the Random Forest (73.2%) performed clearly worse. All differences between the models were significant according to the McNemar test. The difference between the GRU and the 1D-CNN was significant (p = 0.0018), and the difference between the SVM and the Random Forest had p = 0.03.""",
    """On validation, the 1D-CNN was slightly better than the GRU (87.9% compared with 86.6%), so the webcam application uses the 1D-CNN. This choice was not changed after the test results were known, because selecting a model on the test set would make the reported test results too optimistic.""",
]

TABLE_WORDS = [["Model", "Validation accuracy", "Test accuracy (1,753)", "95% CI", "Macro precision", "Macro recall", "Macro F1"],
               ["SVM", "74.4%", "75.4%", "73.4-77.4%", "0.764", "0.747", "0.747"],
               ["Random Forest", "69.8%", "73.2%", "71.1-75.2%", "0.735", "0.725", "0.722"],
               ["1D-CNN", "87.9%", "83.3%", "81.5-85.0%", "0.842", "0.829", "0.831"],
               ["GRU", "86.6%", "85.9%", "84.1-87.4%", "0.862", "0.854", "0.855"]]

RESULTS_WORDS_2 = [
    """Table 5 shows the accuracy for each test signer. The results depend strongly on the signer. The GRU reached 95.4% and 93.6% for two signers, but only 81.9% and 68.9% for the other two. The GRU was the best model for three of the four signers. For signer 30680, the 1D-CNN was better (72.9%), and for signer 37779 the SVM (92.3%) outperformed the 1D-CNN (88.8%).""",
]

TABLE_PEOPLE = [["Model", "Signer 30680 (350 clips)", "Signer 37779 (456 clips)", "Signer 49445 (491 clips)", "Signer 62590 (456 clips)"],
                ["SVM", "59.7%", "92.3%", "59.3%", "87.7%"],
                ["Random Forest", "62.0%", "84.6%", "56.8%", "87.9%"],
                ["1D-CNN", "72.9%", "88.8%", "77.0%", "92.5%"],
                ["GRU", "68.9%", "93.6%", "81.9%", "95.4%"]]

RESULTS_WORDS_3 = [
    """Figure 7 shows the training history of the GRU. The training accuracy reached almost 100%, while the validation accuracy remained at about 86%. The validation loss changed very little after about 60 epochs. This indicates some overfitting, but the validation result did not decrease at the end of training. Figure 8 shows the confusion matrix of the GRU. The most difficult word was go, with an F1-score of 0.63. The most frequent errors were finish predicted as bye (11 clips), like predicted as hungry (9 clips) and wait predicted as why (9 clips). Among the best recognized words were water (0.94), dad (0.93) and tomorrow (0.92).""",
]

RESULTS_CRITERIA = [
    """Table 6 compares all models with the success criteria (accuracy of at least 85% and macro F1-score of at least 0.80). For the letters, no model met both criteria on all 720 test images. The SVM met the macro F1 criterion (0.848) but its accuracy (83.3%) was 1.7 percentage points below the threshold. Most of this gap is caused by the 37 images where MediaPipe did not detect a hand. On the 683 images with a detected hand, the SVM met both criteria with 87.8% accuracy and a macro F1-score of 0.871. For the words, the GRU met both criteria (85.9% and 0.855). The 1D-CNN met the macro F1 criterion (0.831) but not the accuracy criterion (83.3%). On validation data, all four trained letter models would meet both criteria. This again shows why the criteria must be checked on data from new signers.""",
    """The webcam application was also tested manually. The letter mode works in real time with one or two hands. The word mode works for clear signs made in front of the upper body, but the shoulders must be visible. The accuracy of the application was not measured on new recordings, so no number is reported for it.""",
]

TABLE_CRITERIA = [["Task", "Model", "Test data", "Accuracy", "Macro F1", "Successful"],
                  ["Letters", "SVM", "all 720 images", "83.3%", "0.848", "No"],
                  ["Letters", "SVM", "683 images, hand found", "87.8%", "0.871", "Yes"],
                  ["Letters", "ViT", "all 720 images", "79.6%", "0.782", "No"],
                  ["Letters", "Random Forest", "all 720 images", "77.4%", "0.784", "No"],
                  ["Letters", "MobileNetV2, hand crop", "all 720 images", "71.7%", "0.709", "No"],
                  ["Letters", "MobileNetV2, whole image", "all 720 images", "59.0%", "0.550", "No"],
                  ["Words", "GRU", "1,753 clips", "85.9%", "0.855", "Yes"],
                  ["Words", "1D-CNN", "1,753 clips", "83.3%", "0.831", "No"],
                  ["Words", "SVM", "1,753 clips", "75.4%", "0.747", "No"],
                  ["Words", "Random Forest", "1,753 clips", "73.2%", "0.722", "No"]]

DISCUSSION = [
    """The main finding of this project is the large gap between validation and test results. On validation, all letter models reached between 96% and 100%. On the external test, with different signers and backgrounds, the accuracy was between 59% and 83%. If only a random split of the training data had been used, all models would appear almost perfect, and the comparison would provide little information.""",
    """MobileNetV2 illustrates this problem most clearly. It reached 99.9% on validation but only 71.7% on the test, and the version that uses the whole image reached only 59.0%. The training images share the same background and lighting, so the network can use the background as a shortcut, as described by Geirhos et al. (2020). Cropping the hand removed most of the background and increased the test accuracy by 12.7 percentage points. The landmark models do not use the pixels at all, so the background cannot influence their input. This is the most likely reason why a simple SVM on 63 features outperformed a deep network. The learning curve supports this explanation: after about 1,000 images per letter, additional images of the same kind did not improve the results on new signers.""",
    """The pretrained ViT was the second best letter model. It does not depend on MediaPipe, so it can classify every image. However, on the 683 images with a detected hand it was clearly worse than the SVM (79.5% compared with 87.8%). It was trained on the same uniform dataset, so it probably suffers from a similar background problem, although to a smaller degree than our MobileNetV2.""",
    """The success criteria show that the main weakness of the letter system is hand detection rather than classification. The SVM met both criteria when a hand was detected, but the 37 images without a detected hand reduced its accuracy on the full test set to 83.3%. A more reliable hand detector, or a fallback to an image model such as the ViT when no hand is found, could therefore move the system above the 85% threshold.""",
    """The letter N was the most difficult class for almost all models, and it was usually predicted as M. These two signs differ only in the number of fingers placed over the thumb (two for N and three for M). In the landmarks this difference is small, and the fingers often occlude each other. This is a limitation of 2D landmarks from a single camera.""",
    """For the words, the temporal order of the frames was essential. The SVM and the Random Forest only receive an average of the clip over 8 time segments, and they were about 10 percentage points worse than the networks. The GRU and the 1D-CNN learn how the hands move over time, and only the GRU met both success criteria. The accuracy of the GRU ranged from 68.9% to 95.4% across the test signers. This shows that people perform the same sign in different ways, for example faster, smaller or with a different hand position. This agrees with the motivation of signer-independent evaluation in Li et al. (2020). With only 14 training signers, the models cannot observe all of these signing styles. Most errors occurred between signs with a similar movement near the same part of the body, such as finish and bye, or sad, mad and sleep.""",
    """The project has several limitations. The letter training data comes from very similar videos, and the letter test set is small (720 images), so the confidence intervals are about 5 percentage points wide. Only 25 of the 250 signs in the ISLR dataset were used. The word results are based on one split by signer, and a different split could give somewhat different numbers. The webcam application was only tested manually. Future work could include more words, more varied training data for the letters and a measured evaluation of the application on recordings of new users.""",
]

CONCLUSION = [
    """This project developed and compared models for the recognition of American Sign Language from a camera. The work covered 24 static letters and 25 dynamic words, and all models were evaluated on signers that were not seen during training. A model was considered successful if it reached at least 85% accuracy and a macro F1-score of at least 0.80.""",
    """For the letters, an SVM on 63 hand landmark features was the best model. It recognized 83.3% of the 720 external test images with a macro F1-score of 0.848. On the images with a detected hand, it met both success criteria (87.8% and 0.871). It was significantly better than the pretrained Vision Transformer (79.6%), the Random Forest (77.4%) and the fine-tuned MobileNetV2 (71.7% with the hand crop and 59.0% with the whole image). The networks that process pixels learned the background of the uniform training images, while the landmark models were not affected by this problem.""",
    """For the words, models that process the sequence of frames performed much better. The GRU recognized the 25 words of four new signers with 85.9% accuracy and a macro F1-score of 0.855, so it met both success criteria. The 1D-CNN reached 83.3%, while the SVM on a summary of the clip reached 75.4%. The accuracy varied strongly between signers, from 68.9% to 95.4%.""",
    """The project also showed that a validation set from the same source as the training data is not sufficient. Only the evaluation on new signers revealed the real differences between the models. A webcam application was built that recognizes letters and words in real time with the selected models. Future work could add more words, more varied training data and a measured evaluation of the application with new users.""",
]

REFERENCES = [
    "Chow, A., Cameron, G., Georg, M., Sherwood, M., Culliton, P., Sepah, S., Dane, S., & Starner, T. (2023). Google - Isolated sign language recognition [Data set]. Kaggle. https://www.kaggle.com/competitions/asl-signs",
    "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197",
    "Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11), 665-673. https://doi.org/10.1038/s42256-020-00257-z",
    "Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems (2nd ed.). O'Reilly Media.",
    "Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1459-1469). https://doi.org/10.1109/WACV45572.2020.9093512",
    "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4510-4520). https://doi.org/10.1109/CVPR.2018.00474",
    "Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking. arXiv. https://arxiv.org/abs/2006.10214",
]


# ---------------------------------------------------------------- document helpers

def setup(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(1)
        header = s.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = header.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)


def para(doc, text, indent=True, align=None, bold=False, italic=False, keep=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def heading(doc, text, level=1):
    p = para(doc, text, indent=False, bold=True, keep=True,
             align=WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT)
    return p


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def caption(doc, label, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(label + ". ").bold = True
    p.add_run(text).italic = True


def figure(doc, path, number, text, width=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    caption(doc, f"Figure {number}", text)


def table(doc, rows, number, text, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.keep_with_next = True      # table and caption stay on one page
            run = p.add_run(value)
            run.font.size = Pt(10)
            run.bold = i == 0
            if widths:
                cell.width = Inches(widths[j])
        tr_pr = t.rows[i]._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
    caption(doc, f"Table {number}", text)


def build():
    FIG.mkdir(exist_ok=True)
    if not (FIG / "report_examples.png").exists():
        figure_examples(FIG / "report_examples.png")
    if not (FIG / "report_word_hello.png").exists():
        figure_word(FIG / "report_word_hello.png")

    doc = Document()
    setup(doc)

    # title page
    for _ in range(3):
        doc.add_paragraph()
    para(doc, TITLE, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    doc.add_paragraph()
    for line in (f"Student Names: {AUTHORS}", "Astana IT University", COURSE,
                 f"Instructor: {INSTRUCTOR}", f"Academic Year: {YEAR}"):
        para(doc, line, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    page_break(doc)

    heading(doc, "Abstract")
    para(doc, ABSTRACT, indent=False)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0.5)
    p.add_run("Keywords: ").italic = True
    p.add_run(KEYWORDS)
    page_break(doc)

    heading(doc, TITLE)
    heading(doc, "Introduction")
    for t in INTRODUCTION:
        para(doc, t)

    heading(doc, "Literature Review")
    for t in LITERATURE:
        para(doc, t)

    heading(doc, "Dataset and Preprocessing")
    para(doc, DATA[0])
    table(doc, TABLE_DATA, 1, "Datasets Used in the Project", widths=[2.2, 0.7, 1.6, 1.9])
    figure(doc, FIG / "report_examples.png", 1,
           "Training and External Test Images of the Letters A and N with the MediaPipe Hand Landmarks")
    for t in DATA[1:5]:
        para(doc, t)
    figure(doc, FIG / "report_word_hello.png", 2,
           "Four Frames of a Kaggle ISLR Clip of the Word Hello, Drawn from Its Landmarks")
    for t in DATA[5:]:
        para(doc, t)

    heading(doc, "Methodology")
    for t in METHODS:
        para(doc, t)

    heading(doc, "Experiments and Results")
    heading(doc, "Letters", level=2)
    para(doc, RESULTS_LETTERS[0])
    table(doc, TABLE_LETTERS, 2, "Results of the Letter Models on the External Test Set",
          widths=[1.6, 0.8, 0.9, 0.9, 0.75, 0.7, 0.65])
    for t in RESULTS_LETTERS[1:]:
        para(doc, t)
    figure(doc, FIG / "learning_curve_landmarks.png", 3,
           "Learning Curves of the SVM and the Random Forest on the Landmark Features")
    figure(doc, FIG / "learning_curve_mobilenet_hand.png", 4,
           "Loss and Accuracy of MobileNetV2 with the Hand Crop for Each Epoch")
    figure(doc, FIG / "model_comparison.png", 5,
           "Test Accuracy of the Letter Models with 95% Bootstrap Confidence Intervals")
    para(doc, RESULTS_LETTERS_2[0])
    table(doc, TABLE_MCNEMAR, 3, "Exact McNemar Test for the Main Pairs of Letter Models (All 720 Images Unless Noted)",
          widths=[1.7, 2.0, 0.9, 0.9, 0.9])
    para(doc, RESULTS_LETTERS_3[0])
    figure(doc, FIG / "confusion_svm.png", 6, "Confusion Matrix of the SVM on the External Test Set", width=4.8)

    heading(doc, "Words", level=2)
    para(doc, RESULTS_WORDS[0])
    table(doc, TABLE_WORDS, 4, "Results of the Word Models on the Four Test Signers",
          widths=[1.2, 0.9, 0.9, 1.0, 0.75, 0.7, 0.65])
    para(doc, RESULTS_WORDS[1])
    para(doc, RESULTS_WORDS_2[0])
    table(doc, TABLE_PEOPLE, 5, "Test Accuracy of the Word Models for Each Test Signer",
          widths=[1.3, 1.3, 1.3, 1.3, 1.3])
    para(doc, RESULTS_WORDS_3[0])
    figure(doc, FIG / "learning_curve_words_gru.png", 7, "Loss and Accuracy of the GRU for Each Epoch")
    figure(doc, FIG / "words_confusion_gru.png", 8, "Confusion Matrix of the GRU on the Four Test Signers", width=5.0)

    heading(doc, "Success Criteria", level=2)
    para(doc, RESULTS_CRITERIA[0])
    table(doc, TABLE_CRITERIA, 6, "Comparison of the Models with the Success Criteria (Accuracy of at Least 85% and Macro F1 of at Least 0.80)",
          widths=[0.7, 1.8, 1.7, 0.8, 0.8, 0.9])
    para(doc, RESULTS_CRITERIA[1])

    heading(doc, "Discussion")
    for t in DISCUSSION:
        para(doc, t)

    heading(doc, "Conclusion")
    for t in CONCLUSION:
        para(doc, t)

    page_break(doc)
    heading(doc, "References")
    for ref in REFERENCES:
        p = para(doc, ref, indent=False)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)

    doc.save(OUT)
    print(f"Saved {OUT}")


def to_pdf():
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        d = word.Documents.Open(str(OUT.resolve()))
        d.SaveAs(str(OUT.with_suffix(".pdf").resolve()), FileFormat=17)
        d.Close(False)
    finally:
        word.Quit()
    print(f"Saved {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        to_pdf()
