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

ABSTRACT = """This project studies the recognition of American Sign Language (ASL) from a camera. We worked on two tasks: 24 static letters from images and 25 words with movement from video. For the letters, we used a Support Vector Machine (SVM) and a Random Forest on 21 hand landmarks found by MediaPipe. We compared them with MobileNetV2 fine-tuned on images and a ready Vision Transformer (ViT) from Hugging Face. The letter models were trained on the Kaggle ASL Alphabet dataset and tested on a separate set of 720 images with other people and backgrounds. For the words, we used landmark clips from the Kaggle Isolated Sign Language Recognition dataset and compared SVM and Random Forest with a 1D convolutional network (1D-CNN) and a GRU. The word data was split by person, so the 4 test people were never seen in training. All models were evaluated with accuracy, macro precision, recall and F1-score, bootstrap confidence intervals and the McNemar test. The SVM on landmarks was the best letter model with 83.3% accuracy. MobileNetV2 reached 99.9% on validation but only 71.7% on the external test, probably because it also learned the background of the training photos. For the words, the GRU reached 85.9% accuracy and the SVM 75.4%. We also built a webcam application that recognizes letters and words live."""

KEYWORDS = "sign language recognition, MediaPipe, hand landmarks, SVM, MobileNetV2, GRU, McNemar test"

INTRODUCTION = [
    """Sign language is the main language for many deaf and hard of hearing people. Most hearing people do not know it, so everyday communication can be difficult. A program that recognizes signs from a normal webcam could help in simple situations, for example in learning the alphabet or in short conversations. This is why sign language recognition is an active topic in computer vision and machine learning.""",
    """Signs are hard to recognize for a computer. The same letter can look different with another hand, another person, another background or another light. Many signs also have movement, so one image is not enough and the model has to look at a sequence of frames. Another problem is that public datasets are often recorded in very similar conditions. A model can get a very high score on its own dataset and still fail on new people.""",
    """In this project we built a system for American Sign Language (ASL) with two parts. The first part recognizes 24 static letters of the ASL alphabet from images (J and Z are not included because they need movement). The second part recognizes 25 everyday words with movement, such as hello, thank you, water and tomorrow. For the letters we compared classical machine learning on hand landmarks (SVM and Random Forest) with a fine-tuned convolutional network (MobileNetV2) and a ready Vision Transformer. For the words we compared SVM and Random Forest with two neural networks for sequences, a 1D-CNN and a GRU.""",
    """The main goal was to find out which approach works best for people that the model has never seen. For this reason all final results are measured on data from new people: an external test set for the letters and 4 held-out people for the words. We also used statistical tests to check if the differences between models are real. Finally, we built a webcam application that uses the best models to recognize letters and words live.""",
]

LITERATURE = [
    """Sign language recognition is a classification problem: the input is an image or a video, and the output is a letter or a word. Géron (2019) describes the usual steps of such a machine learning project. The data is prepared, a separate test set is kept aside, several models are trained, and they are compared with metrics such as precision, recall and F1-score. We followed these steps in our project.""",
    """Classical models are a good first choice when the input is a small vector of numbers. Support Vector Machines with an RBF kernel and Random Forests are fast, need little tuning and work well on such data (Géron, 2019). A raw image, however, is not a small vector. A hand detector can solve this problem. MediaPipe Hands (Zhang et al., 2020) finds 21 points of the hand in real time on a normal CPU, so a hand image can be turned into 63 numbers.""",
    """The other option is to learn directly from the pixels with a convolutional neural network (CNN). Training a CNN from zero needs a lot of data, so it is common to start from a network that was already trained on ImageNet and fine-tune it on the new task (Géron, 2019). MobileNetV2 (Sandler et al., 2018) is a small and fast CNN that is often used in this way. Many ready image models are also shared on Hugging Face. For example, we tested a Vision Transformer that its author trained on the ASL alphabet.""",
    """For signs with movement, one image is not enough. Recurrent networks such as LSTM and GRU read a sequence step by step and keep a memory of the earlier steps. A 1D convolutional network looks for short patterns in time and is often a fast alternative to a recurrent network (Géron, 2019). Google published the Isolated Sign Language Recognition dataset on Kaggle (Chow et al., 2023), where every video is already given as MediaPipe landmarks. This makes it possible to train such sequence models without processing raw video.""",
    """When two models are tested on the same test set, a difference in accuracy can be random. Dietterich (1998) compared several statistical tests for this case and recommended the McNemar test. We used this test to check the differences between our models.""",
]

DATA = [
    """For the letters we used two Kaggle datasets (Table 1). The ASL Alphabet dataset on Kaggle has 3,000 images for each letter. The images are frames of videos of one hand in front of the same background. We used the 24 static letters, which gives 72,000 training images. For the final test we used the ASL Alphabet Test dataset, also from Kaggle. It has 30 images for each letter (720 in total) with other people, other hands and other backgrounds. Figure 1 shows that the two datasets look very different. The test set was used only for the final results and never to choose a model or a setting.""",
    """The training images are frames of videos, so neighbouring images look almost the same. If we split them at random, almost the same picture would be in the training and the validation part, and the validation score would be too high. So we split the frames in blocks of 50 frames in a row. Every fifth block went to validation and the other blocks to training, which gives a 80/20 split. All letter models used the same split.""",
    """We ran MediaPipe Hands (Zhang et al., 2020) on every image. It found a hand in 55,368 of the 72,000 training images (76.9%) and in 683 of the 720 test images. From the 21 points of the hand we made 63 features (x, y and z of each point). We moved the wrist to the point (0, 0), divided all coordinates by the size of the hand and mirrored the left hand. After this the features do not depend on the position of the hand in the image, on its size or on which hand is used. Images without a found hand were left out of the training of the landmark models. In the test they count as mistakes, because the model gives no answer.""",
    """For the words we used the Google Isolated Sign Language Recognition dataset (Chow et al., 2023). It has 94,477 short videos of 250 ASL signs made by 21 people. The videos are not given as images but as MediaPipe Holistic landmarks for every frame: 468 face points, 21 points for each hand and 33 body pose points. We chose 25 everyday words: hello, bye, thank you, please, yes, no, drink, water, food, hungry, sleep, happy, sad, mad, mom, dad, home, go, wait, look, finish, like, where, why and tomorrow. This gives 9,596 clips, between 346 and 414 for each word. The median clip has only 21 frames. We kept the two hands and the body pose (75 points) and dropped the face. In 326 clips no hand or no shoulders were found, so 9,270 clips were used.""",
    """The word clips were split by person with a fixed random seed. 14 people went to training (6,207 clips), 3 people to validation (1,310 clips) and 4 people to the test (1,753 clips). In this way the test shows how the models work for people they have never seen. Figure 2 shows one clip of the word hello.""",
    """Every clip was prepared in the same way. First, the middle of the shoulders was moved to (0, 0) and all points were divided by the shoulder width. Then the position of the person in the frame and the distance to the camera do not matter. If the left hand was seen in more frames than the right hand, the clip was mirrored, so signs of left-handed people look like signs of right-handed people. Then the clip was cut to the frames where a hand is seen and resampled to 32 time steps. For each step we kept 104 numbers. For each hand these are the wrist position, the hand shape (21 points relative to the wrist) and a flag that shows if the hand is seen. The last numbers are 7 upper body points.""",
    """To get more training data, we made random changes to the training clips. We used a vertical stretch from 0.85 to 1.2, a rotation of up to 0.15 radians, a scale from 0.9 to 1.1 and a small shift. We also cut up to 10% of the frames at the start and at the end. Each training clip had 8 such copies.""",
]

TABLE_DATA = [["Dataset", "Task", "Use", "Size"],
              ["ASL Alphabet (Kaggle)", "letters", "training and validation", "72,000 images, 24 letters"],
              ["ASL Alphabet Test (Kaggle)", "letters", "external test", "720 images, 24 letters"],
              ["Isolated Sign Language Recognition (Kaggle)", "words", "training, validation, test", "9,596 clips, 25 words, 21 people"]]

METHODS = [
    """For the letters we trained and tested five models. The first two use the 63 landmark features. The SVM has an RBF kernel with C = 10, and the features are standardized before training. The Random Forest has 300 trees. Both models were trained with Scikit-learn on all 44,772 training images with a hand. To see how the amount of data changes the result, we also trained both models on 50, 100, 250, 500, 1,000, 1,500 and 2,000 images per letter and made a learning curve.""",
    """The third and fourth models are MobileNetV2 networks (Sandler et al., 2018) that start from ImageNet weights and are fine-tuned with PyTorch, as described by Géron (2019). One network sees the whole image. The other one sees a square around the hand that MediaPipe found, 1.4 times bigger than the hand. The input size is 224 x 224 pixels. First only the last layer was trained for 2 epochs (learning rate 0.001), then the whole network for 8 epochs (learning rate 0.0003). We used the AdamW optimizer, a cosine learning rate schedule and label smoothing of 0.1. The augmentation was strong: random crop, flip, rotation, color changes and random erasing. The epoch with the best validation macro F1 was kept. Training one network took about 45 minutes on an NVIDIA GeForce RTX 4060 Laptop GPU.""",
    """The fifth model is a ready Vision Transformer from Hugging Face, dima806/asl_alphabet_image_detection. Its author trained it on the same ASL Alphabet dataset with 29 classes. We did not train it. We only used the outputs of our 24 letters. This model works on the whole image and does not need MediaPipe, so it can answer for all 720 test images.""",
    """For the words we trained four models. SVM (RBF kernel, C = 10) and Random Forest (500 trees) cannot use a sequence directly, so they get a summary of the clip. The summary is the 32 x 104 sequence averaged in 8 parts of time, plus the mean and the standard deviation over the whole clip (1,040 numbers). They were trained on the original clips and 2 augmented copies. The 1D-CNN has four convolution blocks (kernel size 5, 192 channels, batch normalization, GELU and dropout 0.3) and a pooling over time that takes the mean and the maximum. The GRU first maps every step to 128 numbers and then uses a two-layer bidirectional GRU with 128 units. Both networks were trained for 80 epochs with AdamW, a one-cycle learning rate with the maximum 0.002, batch size 64 and label smoothing of 0.1. In every epoch they saw a new random augmentation. The epoch with the best validation accuracy was kept, and the better network on validation is used in the webcam application.""",
    """All models were compared on the same test data. We report accuracy, macro precision, macro recall and macro F1-score (Géron, 2019), so every letter or word has the same weight. For the letters we report accuracy on all 720 images and on the 683 images where a hand was found. A 95% confidence interval for the accuracy was computed with 2,000 bootstrap samples of the test set. To check if one model is really better than another, we used the exact McNemar test (Dietterich, 1998). It only looks at the test items where one model is right and the other is wrong. We also made confusion matrices and lists of the most common mistakes. All numbers were computed by one script from the saved predictions of every model.""",
    """Finally, we built a webcam application with OpenCV. In the letters mode, MediaPipe finds one or two hands in every frame and the SVM recognizes the letter when the hand is held still for about 1.5 seconds. In the words mode, MediaPipe also finds the body pose. The application records a sign from the moment a hand appears until the hands are lowered or stop moving, and then the word network gives its 3 best guesses.""",
]

RESULTS_LETTERS = [
    """Table 2 shows the results of the five letter models on the external test set. The SVM on landmarks was the best model in every column. It recognized 600 of the 720 test images (83.3%) and 87.8% of the images where MediaPipe found a hand. The ready ViT was second with 79.6%, and the Random Forest third with 77.4%. Both MobileNetV2 models were the weakest: 71.7% for the hand crop and 59.0% for the whole image. Figure 5 shows the same results with the confidence intervals.""",
    """On validation the picture was very different. The SVM reached 98.0% and the Random Forest 96.5%. MobileNetV2 reached 99.9% with the hand crop and 100.0% with the whole image. So every model lost a lot on new people, but MobileNetV2 lost the most. Figure 4 shows the training history of MobileNetV2 with the hand crop. Its validation accuracy is above 99% from the third epoch, so validation could not show the problem.""",
    """Figure 3 shows the learning curve of the landmark models. With only 50 images per letter, the SVM already got 94.6% on validation and 82.3% on the test images with a hand. With 1,000 images per letter it got 97.8% on validation and 87.8% on the test. After that the validation accuracy grew a little more, but the test accuracy did not change. The Random Forest showed the same pattern at a lower level (from 74.1% to about 81.6% on the test).""",
]

TABLE_LETTERS = [["Model", "Accuracy (720)", "95% CI", "Accuracy, hand found (683)", "Macro precision", "Macro recall", "Macro F1"],
                 ["SVM (landmarks)", "83.3%", "80.6-85.8%", "87.8%", "0.890", "0.833", "0.848"],
                 ["ViT (ready model)", "79.6%", "76.7-82.5%", "79.5%", "0.837", "0.796", "0.782"],
                 ["Random Forest (landmarks)", "77.4%", "74.3-80.4%", "81.6%", "0.847", "0.774", "0.784"],
                 ["MobileNetV2, hand crop", "71.7%", "68.3-74.9%", "75.5%", "0.778", "0.717", "0.709"],
                 ["MobileNetV2, whole image", "59.0%", "55.4-62.5%", "58.7%", "0.682", "0.590", "0.550"]]

RESULTS_LETTERS_2 = [
    """Table 3 shows the McNemar test for the most important pairs. The SVM was significantly better than the Random Forest and both MobileNetV2 models (p < 0.0001). Against the ViT, the difference was smaller but still significant: p = 0.04 on all 720 images and p < 0.0001 on the 683 images with a hand. The Random Forest and the ViT were not significantly different (p = 0.26). The hand crop made MobileNetV2 significantly better than the whole image version (p < 0.0001).""",
]

TABLE_MCNEMAR = [["Model A", "Model B", "Only A right", "Only B right", "p-value"],
                 ["SVM", "Random Forest", "62", "19", "< 0.0001"],
                 ["SVM", "ViT", "94", "67", "0.04"],
                 ["SVM", "ViT (hand found)", "94", "37", "< 0.0001"],
                 ["SVM", "MobileNetV2, hand crop", "115", "31", "< 0.0001"],
                 ["Random Forest", "ViT", "80", "96", "0.26"],
                 ["MobileNetV2, hand crop", "MobileNetV2, whole image", "143", "52", "< 0.0001"]]

RESULTS_LETTERS_3 = [
    """Figure 6 shows the confusion matrix of the SVM. Most letters are recognized well, and the F1-score is 1.00 for A, F and Y. The hardest letter is N with an F1-score of 0.43. The most common mistake of the SVM is N predicted as M (18 of 30 images). It is also the top mistake of the Random Forest, the ViT and MobileNetV2 with the hand crop. Other common mistakes of the SVM are C predicted as O (12 images) and S predicted as E (7 images). For 37 test images MediaPipe did not find a hand, and 15 of them are the letters I (10 images) and O (5 images).""",
]

RESULTS_WORDS = [
    """Table 4 shows the results of the four word models on the 1,753 clips of the 4 test people. The GRU was the best model with 85.9% accuracy and a macro F1 of 0.855. The 1D-CNN was close with 83.3%. The SVM (75.4%) and the Random Forest (73.2%) were clearly worse. All differences between the models were significant in the McNemar test. Even the difference between the GRU and the 1D-CNN was significant (p = 0.0018), and the difference between the SVM and the Random Forest had p = 0.03.""",
    """On validation the 1D-CNN was a little better than the GRU (87.9% against 86.6%), so the webcam application uses the 1D-CNN. We did not change this choice after we saw the test results, because choosing a model by the test set would make the test numbers too optimistic.""",
]

TABLE_WORDS = [["Model", "Validation accuracy", "Test accuracy (1,753)", "95% CI", "Macro precision", "Macro recall", "Macro F1"],
               ["SVM", "74.4%", "75.4%", "73.4-77.4%", "0.764", "0.747", "0.747"],
               ["Random Forest", "69.8%", "73.2%", "71.1-75.2%", "0.735", "0.725", "0.722"],
               ["1D-CNN", "87.9%", "83.3%", "81.5-85.0%", "0.842", "0.829", "0.831"],
               ["GRU", "86.6%", "85.9%", "84.1-87.4%", "0.862", "0.854", "0.855"]]

RESULTS_WORDS_2 = [
    """Table 5 shows the accuracy for each test person. The results depend a lot on the person. The GRU got 95.4% and 93.6% for two people, but only 81.9% and 68.9% for the other two. The GRU was the best model for three of the four people. For person 30680 the 1D-CNN was better (72.9%), and for person 37779 even the SVM (92.3%) was better than the 1D-CNN (88.8%).""",
]

TABLE_PEOPLE = [["Model", "Person 30680 (350 clips)", "Person 37779 (456 clips)", "Person 49445 (491 clips)", "Person 62590 (456 clips)"],
                ["SVM", "59.7%", "92.3%", "59.3%", "87.7%"],
                ["Random Forest", "62.0%", "84.6%", "56.8%", "87.9%"],
                ["1D-CNN", "72.9%", "88.8%", "77.0%", "92.5%"],
                ["GRU", "68.9%", "93.6%", "81.9%", "95.4%"]]

RESULTS_WORDS_3 = [
    """Figure 7 shows the training history of the GRU. The training accuracy reached almost 100%, while the validation accuracy stayed at about 86%. The validation loss changed very little after about 60 epochs. So there is some overfitting, but the validation result did not get worse at the end of training. Figure 8 shows the confusion matrix of the GRU. The hardest word was go (F1-score 0.63). The most common mistakes were finish predicted as bye (11 clips), like predicted as hungry (9 clips) and wait predicted as why (9 clips). The best words were water (0.94), dad (0.93) and tomorrow (0.92).""",
    """We also tested the webcam application by hand. The letters mode works in real time with one or two hands. The words mode works for clear signs made in front of the upper body, but it needs the shoulders to be visible. We did not measure its accuracy with new recordings, so we do not report a number for it.""",
]

DISCUSSION = [
    """The main result of this project is the large gap between validation and test results. On validation every letter model got between 96% and 100%. On the external test, with other people and backgrounds, the accuracy was between 59% and 83%. If we had only used a random split of the training data, all models would look almost perfect and the comparison would tell us very little.""",
    """MobileNetV2 shows this problem most clearly. It reached 99.9% on validation but only 71.7% on the test, and the version that sees the whole image got only 59.0%. The training photos have the same background and light, so the network can use the background as a shortcut. Cutting out the hand removed most of the background, and the test accuracy went up by 12.7 percentage points. The landmark models do not see the pixels at all, so the background cannot change their input. We think this is the main reason why a simple SVM on 63 numbers was better than a deep network. The learning curve agrees with this. After about 1,000 images per letter, more images of the same kind did not help on new people.""",
    """The ready ViT was the second best letter model. It does not need MediaPipe, so it can answer for every image. On the 683 images with a hand, however, it was clearly worse than the SVM (79.5% against 87.8%). It was trained on the same uniform dataset, so it probably has a similar background problem, but a large pretrained transformer seems to suffer less than our MobileNetV2.""",
    """The letter N was the hardest for almost all models, and it was usually predicted as M. These two signs differ only in the number of fingers over the thumb (two for N, three for M). In the landmarks this difference is small, and the fingers often cover each other. This is a limit of 2D landmarks from one camera.""",
    """For the words, the order of the frames was important. The SVM and the Random Forest only see an average of the clip in 8 parts of time, and they were about 10 percentage points worse than the networks. The GRU and the 1D-CNN learn how the hands move over time. They were close to each other. The difference between the test people (from 68.9% to 95.4% for the GRU) shows that people make the same sign in different ways, for example faster, smaller or with another hand position. With only 14 training people, the models cannot see all these styles. Most mistakes were between signs with a similar movement near the same part of the body, such as finish and bye, or sad, mad and sleep.""",
    """The project has several limitations. The letter training data comes from very similar videos, and the letter test set is small (720 images), so the confidence intervals are about 5 percentage points wide. We used only 25 of the 250 words in the ISLR dataset. Our split of the word data is only one split by person, and other splits could give somewhat different numbers. The webcam application was only checked by hand. In the future we would like to add more words and train the letter models on more varied data. We would also like to measure the accuracy of the application on recordings of new users.""",
]

CONCLUSION = [
    """In this project we built and compared models for the recognition of American Sign Language from a camera. We worked with 24 static letters and 25 words with movement, and we tested all models on people that they never saw in training.""",
    """For the letters, a simple SVM on 63 hand landmark features was the best model. It recognized 83.3% of the 720 external test images and 87.8% of the images where a hand was found. It was significantly better than the ready Vision Transformer (79.6%), the Random Forest (77.4%) and the fine-tuned MobileNetV2 (71.7% with the hand crop and 59.0% with the whole image). The networks that see pixels learned the background of the uniform training photos, while the landmark models did not have this problem.""",
    """For the words, the models that see the sequence of frames were much better. The GRU recognized the 25 words of 4 new people with 85.9% accuracy and the 1D-CNN with 83.3%, while the SVM on a summary of the clip got 75.4%. The accuracy was very different for different people, from 68.9% to 95.4%.""",
    """We also learned that a validation set from the same source as the training data is not enough. Only the test on new people showed the real differences between the models. Finally, we built a webcam application that recognizes letters and words live with the best models. Future work could add more words, more varied training data and a measured test of the application with new users.""",
]

REFERENCES = [
    "Chow, A., Cameron, G., Georg, M., Sherwood, M., Culliton, P., Sepah, S., Dane, S., & Starner, T. (2023). Google - Isolated sign language recognition [Data set]. Kaggle. https://www.kaggle.com/competitions/asl-signs",
    "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197",
    "Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems (2nd ed.). O'Reilly Media.",
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
    for t in DATA[:1]:
        para(doc, t)
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
    table(doc, TABLE_WORDS, 4, "Results of the Word Models on the 4 Test People",
          widths=[1.2, 0.9, 0.9, 1.0, 0.75, 0.7, 0.65])
    para(doc, RESULTS_WORDS[1])
    para(doc, RESULTS_WORDS_2[0])
    table(doc, TABLE_PEOPLE, 5, "Test Accuracy of the Word Models for Each Test Person",
          widths=[1.3, 1.3, 1.3, 1.3, 1.3])
    para(doc, RESULTS_WORDS_3[0])
    figure(doc, FIG / "learning_curve_words_gru.png", 7, "Loss and Accuracy of the GRU for Each Epoch")
    figure(doc, FIG / "words_confusion_gru.png", 8, "Confusion Matrix of the GRU on the 4 Test People", width=5.0)
    para(doc, RESULTS_WORDS_3[1])

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
