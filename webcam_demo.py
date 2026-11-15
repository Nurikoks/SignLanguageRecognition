"""Live demo with one or two hands.

Letters mode: show an ASL letter with one or both hands and hold it for about 1.5 s.
Words mode:   make one of the 25 ASL words from words.py (model from train_islr.py), then
              lower your hands or hold them still: the 3 best guesses are shown.
Both modes:   the 6-7 meme (both hands go up and down in turns) shows your friend's photo.

Keys: M switch mode, SPACE add a space, BACKSPACE delete, C clear text, Q or Esc quit.

Usage:
  python webcam_demo.py
  python webcam_demo.py --friend friend.jpg
  python webcam_demo.py --width 1280 --height 720 --fps 60   (if your camera can do it)
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

from camera import FpsCounter, add_camera_args, open_camera, open_window
from features import (HAND_COLORS, SIDES, HandTracker, create_detector, detect_hands, draw_hand,
                      normalize)
from gestures import SIX_SEVEN_SECONDS, FrameBuffer, is_six_seven
from word_live import MIN_PROB as WORD_MIN_PROB
from word_live import WORDS_NET, WordModel, WordRecorder, create_pose_detector, frame_array

HOLD_SECONDS = 1.5   # how long a sign must be held
AGREE = 0.7          # share of frames in that time that must give the same letter
MIN_PROB = 0.5       # ignore frames where the model is unsure
LOST_GRACE = 0.4     # a hand lost for less than this (blur) keeps its letter progress

GUESS_SECONDS = 3.0  # how long the word guesses stay on screen
WORD_NAMES = {"thankyou": "thank you"}

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


def draw_word_guesses(frame, guesses, recording):
    h = frame.shape[0]
    if recording:
        cv2.circle(frame, (30, h - 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "recording", (50, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    for i, (word, prob) in enumerate(guesses):
        color = (0, 220, 0) if i == 0 and prob >= WORD_MIN_PROB else (255, 255, 255)
        cv2.putText(frame, f"{i + 1}. {WORD_NAMES.get(word, word)}  {prob:.0%}",
                    (20, 120 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--friend", default="friend.jpg", help="photo shown for the 6-7 gesture")
    add_camera_args(parser)
    args = parser.parse_args()

    letters_model = joblib.load("model.joblib")
    words_model = WordModel() if Path(WORDS_NET).exists() else None
    if words_model is None:
        print(f"No {WORDS_NET}: words mode is off. Run islr_prepare.py and train_islr.py")
    pose_detector = create_pose_detector() if words_model is not None else None
    recorder, guesses, guesses_until = WordRecorder(), [], 0.0
    photo = load_photo(args.friend)
    detector = create_detector(video=True, num_hands=2)
    cap = open_camera(args)
    fps, window_open = FpsCounter(), False

    hand_tracker = HandTracker()
    trackers = {side: LetterTracker() for side in SIDES}
    last_seen = {side: 0.0 for side in SIDES}
    frames = FrameBuffer(SIX_SEVEN_SECONDS)
    mode, text = "letters", ""
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = detect_hands(detector, rgb, ts, hand_tracker)
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

        # ---------- words: record from the first hand until the hands go down ----------
        if mode == "words" and not showing_photo and words_model is not None:
            clip = recorder.add(now, frame_array(pose_detector, rgb, ts, hands))
            if clip is not None:
                guesses, guesses_until = words_model.top(clip, aspect=h / w), now + GUESS_SECONDS
                if guesses and guesses[0][1] >= WORD_MIN_PROB:
                    word = WORD_NAMES.get(guesses[0][0], guesses[0][0])
                    text = add_word(text, word)
                    speech.put(word)
            draw_word_guesses(frame, guesses if now < guesses_until else [], recorder.recording)
        elif mode == "words" and words_model is None:
            cv2.putText(frame, "no words model: run islr_prepare.py, train_islr.py",
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

        cv2.putText(frame, f"{fps.tick():.0f} fps", (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        if not window_open:
            open_window("SignVision", w, h)
            window_open = True
        cv2.imshow("SignVision", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mode = "words" if mode == "letters" else "letters"
            frames.clear()
            recorder.reset()
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
