"""The words (signs with movement) we recognize, taken from the Kaggle ISLR dataset
(Google Isolated Sign Language Recognition, 250 ASL signs). Names are the dataset's."""
from pathlib import Path

ISLR_DIR = Path("data/islr")

WORDS = [
    "hello", "bye", "thankyou", "please", "yes", "no",
    "drink", "water", "food", "hungry", "sleep",
    "happy", "sad", "mad",
    "mom", "dad", "home",
    "go", "wait", "look", "finish", "like",
    "where", "why", "tomorrow",
]
