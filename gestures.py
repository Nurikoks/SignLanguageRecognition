"""Gestures with movement and two hands: features for word recognition and the 6-7 detector.

A recording is a few seconds of webcam frames, stored as arrays:
  points  T x 2 x 21 x 3   landmarks in pixels (hand 0 = Left, hand 1 = Right)
  present T x 2            True where that hand was found in that frame
"""
from collections import deque
from pathlib import Path

import numpy as np

from features import PALM, SIDES, hand_size, normalize

WORDS_DIR = Path("data/words")
WORDS_MODEL = "words_model.joblib"
NOTHING = "nothing"      # optional class: random moves that are not a word, never shown

RECORD_SECONDS = 2.0     # length of one word example (and of the live window)
TRAJ_STEPS = 16          # the wrist path is resampled to this many points
SHAPE_STEPS = 4          # the hand shape is averaged over this many parts of the gesture
MIN_FRAMES = 8
MIN_PRESENT = 0.3        # a hand seen in fewer frames than this counts as not used

SIX_SEVEN_SECONDS = 1.5  # window for the 6-7 detector


def to_arrays(hands):
    """{'Left': points, 'Right': points} -> (2x21x3 points, 2 presence flags)."""
    points = np.zeros((2, 21, 3), np.float32)
    present = np.zeros(2, bool)
    for i, side in enumerate(SIDES):
        if side in hands:
            points[i] = hands[side]
            present[i] = True
    return points, present


class FrameBuffer:
    """The last few seconds of webcam frames."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.frames = deque()

    def add(self, t, hands):
        self.frames.append((t, *to_arrays(hands)))
        while self.frames and t - self.frames[0][0] > self.seconds:
            self.frames.popleft()

    def clear(self):
        self.frames.clear()

    def last(self, seconds):
        """(times, points, present) for the frames of the last `seconds`."""
        if not self.frames:
            return np.zeros(0), np.zeros((0, 2, 21, 3), np.float32), np.zeros((0, 2), bool)
        end = self.frames[-1][0]
        t, points, present = zip(*[f for f in self.frames if end - f[0] <= seconds])
        return np.array(t), np.stack(points), np.stack(present)


def _fill_gaps(points, present):
    """Where a hand was lost for a few frames, copy the nearest frame where it was seen."""
    points = points.copy()
    used = present.mean(axis=0) >= MIN_PRESENT
    for s in range(2):
        if not used[s]:
            points[:, s] = 0
            continue
        seen = np.flatnonzero(present[:, s])
        for i in np.flatnonzero(~present[:, s]):
            points[i, s] = points[seen[np.abs(seen - i).argmin()], s]
    return points, used


def sequence_features(points, present):
    """Turn a recording into one fixed-size vector (or None if there are no hands):
    which hands are used, the path of each wrist, the vector between the wrists,
    and the hand shapes in SHAPE_STEPS parts of the gesture."""
    if len(points) < MIN_FRAMES:
        return None
    points, used = _fill_gaps(points, present)
    if not used.any():
        return None
    T = len(points)
    scale = np.mean([hand_size(points[i, s]) for i in range(T) for s in range(2) if used[s]])
    if scale <= 0:
        return None

    idx = np.linspace(0, T - 1, TRAJ_STEPS).round().astype(int)
    wrists = points[:, :, 0, :2]
    feats = [used.astype(np.float32)]
    for s in range(2):
        # path around the hand's average position, in hand sizes, so it does not
        # depend on where you stand or how far you are from the camera
        path = (wrists[idx, s] - wrists[:, s].mean(axis=0)) / scale
        feats.append(path.ravel() if used[s] else np.zeros(TRAJ_STEPS * 2))
    between = (wrists[idx, 1] - wrists[idx, 0]) / scale
    feats.append(between.ravel() if used.all() else np.zeros(TRAJ_STEPS * 2))
    for s, side in enumerate(SIDES):
        for part in np.array_split(np.arange(T), SHAPE_STEPS):
            if used[s]:
                feats.append(np.mean([normalize(points[i, s], side) for i in part], axis=0))
            else:
                feats.append(np.zeros(63))
    return np.concatenate(feats).astype(np.float32)


def motion_amount(points, present):
    """How far the most moving wrist traveled, in hand sizes (about 0 for a still hand).
    Uses the size of the area the wrist covered, so camera jitter does not add up."""
    best = 0.0
    for s in range(2):
        seen = np.flatnonzero(present[:, s])
        if len(seen) < 2:
            continue
        scale = np.mean([hand_size(points[i, s]) for i in seen])
        if scale > 0:
            best = max(best, np.linalg.norm(np.ptp(points[seen, s, 0, :2], axis=0)) / scale)
    return best


def _moves(y, step):
    """Count up and down moves longer than `step` (small shakes are ignored)."""
    count, direction, extreme = 0, 0, y[0]
    for v in y[1:]:
        if direction and (v - extreme) * direction > 0:
            extreme = v                      # still going the same way
        elif abs(v - extreme) >= step:
            direction = 1 if v > extreme else -1
            count += 1
            extreme = v
    return count


def is_six_seven(points, present):
    """The 6-7 meme: both hands go up and down in turns, like weighing something
    (one hand goes up while the other goes down), a few times in a row."""
    if len(present) < 15:
        return False
    both = present.all(axis=1)
    if both.mean() < 0.7:
        return False
    p = points[both]
    scale = np.mean([hand_size(hand) for hand in p.reshape(-1, 21, 3)])
    if scale <= 0:
        return False
    kernel = np.ones(3) / 3
    y = [np.convolve(p[:, s, PALM, 1].mean(axis=1), kernel, mode="valid") / scale
         for s in range(2)]
    if min(np.ptp(y[0]), np.ptp(y[1])) < 0.4:      # moves must be big enough
        return False
    if np.corrcoef(y[0], y[1])[0, 1] > -0.4:       # hands must go opposite ways
        return False
    return min(_moves(y[0], 0.2), _moves(y[1], 0.2)) >= 3   # up, down, up at least
