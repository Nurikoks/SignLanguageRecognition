"""Record examples of a word gesture with the webcam. Both hands are tracked.

Usage:
  python record_words.py HELLO --count 20
  python record_words.py nothing --count 20   (optional: random moves that are not a word)

Press SPACE to start or pause, Q or Esc to quit. For each example: get ready while
the countdown runs, then make the gesture while the frame is red (2 s).
Examples are saved to data/words/<WORD>/.
"""
import argparse
import time

import cv2
import numpy as np

from camera import FpsCounter, add_camera_args, open_camera, open_window
from features import HAND_COLORS, HandTracker, create_detector, detect_hands, draw_hand
from gestures import RECORD_SECONDS, WORDS_DIR, to_arrays

READY_SECONDS = 1.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("word", help="name of the word (Latin letters, e.g. HELLO)")
    parser.add_argument("--count", type=int, default=20, help="how many examples to record")
    add_camera_args(parser)
    args = parser.parse_args()

    folder = WORDS_DIR / args.word
    folder.mkdir(parents=True, exist_ok=True)
    before = len(list(folder.glob("*.npz")))
    print(f"{args.word}: {before} examples already recorded")

    detector = create_detector(video=True, num_hands=2)
    cap = open_camera(args)
    fps, window_open = FpsCounter(), False
    tracker = HandTracker()
    state, state_since, frames, saved = "paused", 0.0, [], 0
    start, last_ts = time.time(), -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts
        hands = detect_hands(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts, tracker)
        for side, points in hands.items():
            draw_hand(frame, points, HAND_COLORS[side])

        if state == "ready" and now - state_since >= READY_SECONDS:
            state, state_since, frames = "rec", now, []
        if state == "rec":
            frames.append((now, *to_arrays(hands)))
            if now - state_since >= RECORD_SECONDS:
                t, points, present = (np.stack(x) for x in zip(*frames))
                if present.any(axis=1).mean() < 0.5:
                    print("Hands were not visible, example not saved")
                else:
                    np.savez_compressed(folder / f"{int(now * 1000)}.npz",
                                        t=t, points=points, present=present)
                    saved += 1
                if saved >= args.count:
                    print(f"Done: {saved} new examples in {folder}")
                    state = "paused"
                    saved = 0
                    before = len(list(folder.glob("*.npz")))
                else:
                    state, state_since = "ready", now

        h, w = frame.shape[:2]
        if state == "rec":
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "REC", (w - 110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        elif state == "ready":
            left = READY_SECONDS - (now - state_since)
            cv2.putText(frame, f"{left:.1f}", (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 255, 255), 6)
        else:
            cv2.putText(frame, "SPACE to start", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"{args.word}: {saved}/{args.count}  (total {before + saved})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"{fps.tick():.0f} fps", (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        if not window_open:
            open_window("Record word", w, h)
            window_open = True
        cv2.imshow("Record word", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            state, state_since = ("ready", now) if state == "paused" else ("paused", now)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
