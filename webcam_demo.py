"""Live demo: show an ASL letter to the webcam, hold it for about 1.5 seconds,
and the program shows and says the letter. Press Q or Esc to quit.

Usage:
  python webcam_demo.py
"""
import queue
import threading
import time
from collections import Counter, deque

import cv2
import joblib

from features import HAND_CONNECTIONS, create_detector, detect, normalize

HOLD_SECONDS = 1.5   # how long a sign must be held
AGREE = 0.7          # share of frames in that time that must give the same letter
MIN_PROB = 0.5       # ignore frames where the model is unsure

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


def draw_hand(frame, points):
    for a, b in HAND_CONNECTIONS:
        pa, pb = points[a, :2].astype(int), points[b, :2].astype(int)
        cv2.line(frame, tuple(pa), tuple(pb), (255, 255, 255), 2)
    for p in points[:, :2].astype(int):
        cv2.circle(frame, tuple(p), 4, (0, 140, 255), -1)


def main():
    model = joblib.load("model.joblib")
    detector = create_detector(video=True)
    cap = cv2.VideoCapture(0)
    history = deque()            # (time, letter) for the last HOLD_SECONDS
    shown, last_said = "", ""
    start, last_ts = time.time(), -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # mirror view, feels natural for the user
        now = time.time()
        ts = max(int((now - start) * 1000), last_ts + 1)   # must always increase
        last_ts = ts
        points, side = detect(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)

        progress, candidate = 0.0, ""
        if points is None:
            history.clear()
            last_said = ""           # same letter can be said again after hand leaves
            cv2.putText(frame, "No hand", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            draw_hand(frame, points)
            probs = model.predict_proba([normalize(points, side)])[0]
            if probs.max() >= MIN_PROB:
                history.append((now, model.classes_[probs.argmax()]))
            while history and now - history[0][0] > HOLD_SECONDS:
                history.popleft()
            if history:
                candidate, count = Counter(l for _, l in history).most_common(1)[0]
                if count / len(history) >= AGREE:
                    progress = min((now - history[0][0]) / HOLD_SECONDS, 1.0)
                if progress > 0.9 and candidate != last_said:
                    shown = last_said = candidate
                    speech.put(candidate)

        # progress bar for the current guess
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (80, 80, 80), 2)
        cv2.rectangle(frame, (20, h - 40), (20 + int((w - 40) * progress), h - 20), (0, 200, 0), -1)
        if candidate:
            cv2.putText(frame, f"guess: {candidate}", (20, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # big confirmed letter
        cv2.putText(frame, shown, (w - 120, 110), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 200, 0), 8)

        cv2.imshow("SignVision", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
