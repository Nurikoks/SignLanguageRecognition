"""The 6-7 detector: a buffer of the last webcam frames and the check on it.
(Words are recognized by word_live.py with the model from train_islr.py.)

The buffer keeps a few seconds of frames as arrays:
  points  T x 2 x 21 x 3   landmarks in pixels (hand 0 = Left, hand 1 = Right)
  present T x 2            True where that hand was found in that frame
"""
from collections import deque

import numpy as np

from features import PALM, SIDES, hand_size

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
