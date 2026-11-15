"""Show how each of the 25 words is made: a real Kaggle ISLR clip drawn as a skeleton.

Usage:
  python word_examples.py            # window: N / P next / previous word, S slow, Q quit
  python word_examples.py --save     # also save word_examples/<word>.mp4

For every word we take the clips of the training people and choose a clear one: long
enough, the hands are seen in most frames, and the word model (words_model.pt) recognizes it
with a high probability.
The skeleton is mirrored like the webcam demo window: copy it as if it were a mirror.
Which hand you use does not matter, the model mirrors left-handed signs.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from features import HAND_CONNECTIONS
from train_islr import split_people
from word_features import ISLR_ASPECT
from word_live import WordModel
from words import ISLR_DIR

SIZE = 520
BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
FACE = list(range(11))
CANDIDATES = 60          # clips per word that we score with the model
MIN_FRAMES = 25          # shorter clips are hard to watch
HOLD = 6                 # a hand lost for up to this many frames is drawn where it was
FPS = 30
NAMES = {"thankyou": "thank you"}


def best_clips(model):
    meta = pd.read_csv(ISLR_DIR / "clips.csv")
    _, _, train_people = split_people(meta["participant_id"].unique())
    meta = meta[meta["participant_id"].isin(train_people)]
    best = {}
    for word, g in meta.groupby("sign"):
        top = (-1.0, 0.0, None)
        for sid in g["sequence_id"].sample(min(CANDIDATES, len(g)), random_state=0):
            clip = np.load(ISLR_DIR / "clips" / f"{sid}.npy").astype(np.float32)
            if len(clip) < MIN_FRAMES:
                continue
            seen = np.isfinite(clip[:, [0, 21], 0]).any(axis=1).mean()
            prob = dict(model.top(clip, ISLR_ASPECT, k=len(model.words))).get(word, 0.0)
            if prob * seen > top[0]:
                top = (prob * seen, prob, clip)
        best[word] = top[1:]
        print(f"  {word:10s} p = {top[1]:.2f}  frames {len(top[2])}", flush=True)
    return best


def to_pixels(clip):
    """Clip in image coordinates -> pixels on a SIZE x SIZE canvas (mirrored)."""
    pts = clip[..., :2].copy()
    pts[..., 0] = 1 - pts[..., 0]                    # mirror like the demo window
    pts[..., 1] *= ISLR_ASPECT
    for start in (0, 21):                            # hide short gaps of a hand
        last = None
        for t in range(len(pts)):
            if np.isfinite(pts[t, start, 0]):
                last = t
            elif last is not None and t - last <= HOLD:
                pts[t, start:start + 21] = pts[last, start:start + 21]
    body = pts[:, 42 + np.array(FACE + [11, 12, 13, 14, 15, 16])].reshape(-1, 2)
    allp = np.concatenate([body, pts[:, :42].reshape(-1, 2)])
    allp = allp[np.isfinite(allp).all(axis=1)]
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = 0.85 * SIZE / max(hi - lo)
    center = (lo + hi) / 2
    return (pts - center) * scale + SIZE / 2


def draw(frame_pts, word, prob, i, n, slow):
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    pose = frame_pts[42:]

    def ok(p):
        return np.isfinite(p).all()

    for a, b in BODY:
        if ok(pose[a]) and ok(pose[b]):
            cv2.line(img, tuple(pose[a].astype(int)), tuple(pose[b].astype(int)), (200, 200, 200), 3)
    for k in FACE:
        if ok(pose[k]):
            cv2.circle(img, tuple(pose[k].astype(int)), 4, (180, 180, 180), -1)
    for start, color in ((0, (255, 160, 60)), (21, (60, 160, 255))):
        hand = frame_pts[start:start + 21]
        if not ok(hand[0]):
            continue
        for a, b in HAND_CONNECTIONS:
            if ok(hand[a]) and ok(hand[b]):
                cv2.line(img, tuple(hand[a].astype(int)), tuple(hand[b].astype(int)), color, 2)
        for p in hand:
            if ok(p):
                cv2.circle(img, tuple(p.astype(int)), 3, (255, 255, 255), -1)
    cv2.putText(img, f"{i + 1}/{n}  {NAMES.get(word, word).upper()}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 230, 80), 2)
    cv2.putText(img, f"model p = {prob:.2f}" + ("   SLOW" if slow else ""), (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "N next  P previous  S slow  Q quit", (15, SIZE - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="save word_examples/<word>.mp4")
    args = parser.parse_args()

    print("Choosing the clearest clip for every word...")
    model = WordModel()
    best = best_clips(model)
    words = [w for w in model.words if w in best]
    pixels = {w: to_pixels(best[w][1]) for w in words}

    if args.save:
        out = Path("word_examples")
        out.mkdir(exist_ok=True)
        for w in words:
            writer = cv2.VideoWriter(str(out / f"{w}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     FPS, (SIZE, SIZE))
            for _ in range(2):
                for t, p in enumerate(pixels[w]):
                    writer.write(draw(p, w, best[w][0], words.index(w), len(words), False))
            writer.release()
        print(f"Saved {len(words)} videos to {out}/")

    i, t, slow = 0, 0, False
    cv2.namedWindow("Word examples", cv2.WINDOW_NORMAL)
    while True:
        w = words[i]
        frames = pixels[w]
        cv2.imshow("Word examples", draw(frames[t % len(frames)], w, best[w][0], i, len(words), slow))
        t += 1
        if t % len(frames) == 0:                     # short pause between repeats
            cv2.waitKey(500)
        key = cv2.waitKey(int(1000 / FPS * (3 if slow else 1))) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty("Word examples", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("n"):
            i, t = (i + 1) % len(words), 0
        elif key == ord("p"):
            i, t = (i - 1) % len(words), 0
        elif key == ord("s"):
            slow = not slow
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
