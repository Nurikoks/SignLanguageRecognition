"""Live demo with one or two hands.

Letters mode: show an ASL letter with one or both hands and hold it for about 1.5 s.
Words mode:   make a word gesture recorded with record_words.py (needs words_model.joblib).
Both modes:   the 6-7 meme (both hands go up and down in turns) shows your friend's photo.

Keys: M switch mode, SPACE add a space, BACKSPACE delete, C clear text, Q or Esc quit.

Usage:
  python webcam_demo.py
  python webcam_demo.py --friend friend.jpg
"""
import argparse
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import (HAND_COLORS, SIDES, HandTracker, create_detector, detect_hands, draw_hand,
                      normalize)
from gestures import (NOTHING, RECORD_SECONDS, SIX_SEVEN_SECONDS, WORDS_MODEL,
                      FrameBuffer, is_six_seven, motion_amount, sequence_features)

HOLD_SECONDS = 1.5   # how long a sign must be held
AGREE = 0.7          # share of frames in that time that must give the same letter
MIN_PROB = 0.5       # ignore frames where the model is unsure
LOST_GRACE = 0.4     # a hand lost for less than this (blur) keeps its letter progress

WORD_MIN_PROB = 0.7  # the word model must be at least this sure...
WORD_STREAK = 3      # ...for this many frames in a row
WORD_PAUSE = 1.0     # seconds before the next word can be recognized
MIN_MOTION = 0.4     # hands must move at least this much (in hand sizes) to make a word

PHOTO_SECONDS = 3.0  # how long the 6-7 photo stays on screen

# ---------- text to speech in a background thread, so video does not freeze ----------
speech = queue.Queue()


def speech_worker():
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 not installed, letters will only be shown on screen")
        return
    while True:
        text = speech.get()
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


threading.Thread(target=speech_worker, daemon=True).start()


class LetterTracker:
    """Hold logic for one hand: a letter is accepted when it stays the main guess
    for HOLD_SECONDS."""

    def __init__(self):
        self.history = deque()   # (time, letter) for the last HOLD_SECONDS
        self.last_said = ""

    def reset(self):
        self.history.clear()
        self.last_said = ""      # same letter can be said again after hand leaves

    def update(self, now, letter):
        """letter: the model's guess for this frame, or None if unsure.
        Returns (candidate, progress 0..1, confirmed letter or None)."""
        if letter:
            self.history.append((now, letter))
        while self.history and now - self.history[0][0] > HOLD_SECONDS:
            self.history.popleft()
        if not self.history:
            return "", 0.0, None
        candidate, count = Counter(l for _, l in self.history).most_common(1)[0]
        progress = 0.0
        if count / len(self.history) >= AGREE:
            progress = min((now - self.history[0][0]) / HOLD_SECONDS, 1.0)
        if progress > 0.9 and candidate != self.last_said:
            self.last_said = candidate
            return candidate, progress, candidate
        return candidate, progress, None


def load_photo(path):
    if not Path(path).exists():
        print(f"No friend photo at '{path}': 6-7 will only show text. Use --friend <file>")
        return None
    # imdecode instead of imread, so paths with non-Latin letters also work on Windows
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def draw_six_seven(frame, photo, elapsed):
    h, w = frame.shape[:2]
    if photo is not None:
        grow = 0.5 + 0.5 * min(elapsed / 0.25, 1.0)          # small "pop" at the start
        k = h * 0.6 * grow / max(photo.shape[:2])
        img = cv2.resize(photo, (max(1, int(photo.shape[1] * k)), max(1, int(photo.shape[0] * k))))
        img = cv2.copyMakeBorder(img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        ih, iw = img.shape[:2]
        y, x = (h - ih) // 2 + 30, (w - iw) // 2
        frame[y:y + ih, x:x + iw] = img
    cv2.putText(frame, "6 7 !!!", (w // 2 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0, 0, 0), 14)
    cv2.putText(frame, "6 7 !!!", (w // 2 - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                (0, 255, 255), 6)


def draw_letter_guess(frame, points, candidate, progress):
    """Guess and hold progress above the hand."""
    x, y = int(points[:, 0].min()), int(points[:, 1].min()) - 15
    x, y = max(x, 5), max(y, 60)
    cv2.putText(frame, candidate, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.rectangle(frame, (x, y - 6), (x + 100, y + 4), (80, 80, 80), 1)
    cv2.rectangle(frame, (x, y - 6), (x + int(100 * progress), y + 4), (0, 200, 0), -1)


def add_word(text, word):
    return text + ("" if not text or text.endswith(" ") else " ") + word + " "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--friend", default="friend.jpg", help="photo shown for the 6-7 gesture")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    letters_model = joblib.load("model.joblib")
    words_model = joblib.load(WORDS_MODEL) if Path(WORDS_MODEL).exists() else None
    if words_model is None:
        print(f"No {WORDS_MODEL}: words mode is off. Run record_words.py and train_words.py")
    photo = load_photo(args.friend)
    detector = create_detector(video=True, num_hands=2)
    cap = cv2.VideoCapture(args.camera)

    hand_tracker = HandTracker()
    trackers = {side: LetterTracker() for side in SIDES}
    last_seen = {side: 0.0 for side in SIDES}
    frames = FrameBuffer(max(RECORD_SECONDS, SIX_SEVEN_SECONDS))
    mode, text = "letters", ""
    word, word_prob, last_word, streak, word_ready = None, 0.0, None, 0, 0.0
    six_seven_since = -PHOTO_SECONDS
    start, last_ts = time.time(), -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # mirror view, feels natural for the user
        now = time.time()
        ts = max(int((now - start) * 1000), last_ts + 1)   # must always increase
        last_ts = ts
        hands = detect_hands(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts, hand_tracker)
        frames.add(now, hands)
        for side in hands:
            last_seen[side] = now
        for side, points in hands.items():
            draw_hand(frame, points, HAND_COLORS[side])
        h, w = frame.shape[:2]

        # ---------- 6-7 works in both modes ----------
        showing_photo = now - six_seven_since < PHOTO_SECONDS
        if not showing_photo and is_six_seven(*frames.last(SIX_SEVEN_SECONDS)[1:]):
            six_seven_since, showing_photo = now, True
            speech.put("six seven")
            frames.clear()
            for tracker in trackers.values():
                tracker.reset()

        # ---------- letters: each hand on its own ----------
        if mode == "letters" and not showing_photo:
            for side in SIDES:
                if side not in hands:
                    if now - last_seen[side] > LOST_GRACE:
                        trackers[side].reset()
                    continue
                probs = letters_model.predict_proba([normalize(hands[side], side)])[0]
                letter = letters_model.classes_[probs.argmax()] if probs.max() >= MIN_PROB else None
                candidate, progress, confirmed = trackers[side].update(now, letter)
                if confirmed:
                    text += confirmed
                    speech.put(confirmed)
                draw_letter_guess(frame, hands[side], candidate, progress)

        # ---------- words: the last RECORD_SECONDS of movement ----------
        if mode == "words" and not showing_photo and words_model is not None:
            word, word_prob = None, 0.0
            t, points, present = frames.last(RECORD_SECONDS)
            if (now >= word_ready and len(t) and t[-1] - t[0] >= 0.9 * RECORD_SECONDS
                    and motion_amount(points, present) >= MIN_MOTION):
                x = sequence_features(points, present)
                if x is not None:
                    probs = words_model.predict_proba([x])[0]
                    word, word_prob = words_model.classes_[probs.argmax()], probs.max()
            if word and word != NOTHING and word_prob >= WORD_MIN_PROB:
                streak = streak + 1 if word == last_word else 1
            else:
                streak = 0
            last_word = word
            if streak >= WORD_STREAK:
                text = add_word(text, word)
                speech.put(word)
                frames.clear()
                word_ready, streak = now + WORD_PAUSE, 0
            if word and word != NOTHING:
                cv2.putText(frame, f"guess: {word} {word_prob:.2f}", (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if now < word_ready:
                cv2.putText(frame, "got it!", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 0), 2)
        elif mode == "words" and words_model is None:
            cv2.putText(frame, "no words model: run record_words.py, train_words.py",
                        (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if showing_photo:
            draw_six_seven(frame, photo, now - six_seven_since)

        # text bar and mode
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(frame, text[-38:], (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{mode.upper()} (M)", (w - 170, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        if not hands:
            cv2.putText(frame, "No hands", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("SignVision", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mode = "words" if mode == "letters" else "letters"
            frames.clear()
            streak = 0
            for tracker in trackers.values():
                tracker.reset()
        elif key == ord(" "):
            text += " "
        elif key == 8:               # backspace
            text = text[:-1]
        elif key == ord("c"):
            text = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
