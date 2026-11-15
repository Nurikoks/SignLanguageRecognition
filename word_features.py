"""Features of a word clip (a sign with movement) for the word models.

A clip is an array frames x 75 x 3 with MediaPipe landmarks in image coordinates
(x divided by the image width, y by the height): 0-20 one hand, 21-41 the other hand,
42-74 the pose (33 points). NaN where a part was not found. The Kaggle ISLR clips are
stored like this (islr_prepare.py) and the webcam demo builds the same array.

Steps:
1. y is multiplied by the image height / width, so x and y have the same unit.
2. Everything is moved and scaled so the middle of the shoulders is (0, 0) and the
   shoulder width is 1. Then the position of the person in the frame and the distance
   to the camera do not matter.
3. The hand that is seen in more frames is the "main" hand. If it is the left one, the
   clip is mirrored, so signs by left-handed and right-handed people look the same.
4. The clip is resampled to STEPS frames in time (signs are slow or fast).

Per frame we keep (FRAME_DIM numbers): for each hand its wrist position, its 21 points
relative to the wrist (scaled by the hand size, this is the hand shape) and a flag
"hand seen"; and 7 upper body points (nose, shoulders, elbows, wrists).
"""
import numpy as np

STEPS = 32
POSE = 42
UPPER = [0, 11, 12, 13, 14, 15, 16]        # nose, shoulders, elbows, wrists
L_SHOULDER, R_SHOULDER = 11, 12
FRAME_DIM = 2 * (2 + 42 + 1) + 2 * len(UPPER)
ISLR_ASPECT = 1.2    # estimated from the data: height / width of the Kaggle videos


def _fill(values, ok):
    """Fill frames where ok is False with the nearest frame where it is True."""
    if ok.all() or not ok.any():
        return values
    idx = np.where(ok, np.arange(len(ok)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(ok)
    idx[:first] = first
    return values[idx]


def normalize_clip(clip, aspect, augment=None):
    """clip: T x 75 x 3 -> (T x 75 x 2 in shoulder units, which hand is main, seen flags)."""
    pts = clip[..., :2].astype(np.float32).copy()
    pts[..., 1] *= aspect
    if augment is not None:   # (y stretch, rotation in radians, scale, shift)
        stretch, angle, scale, shift = augment
        pts[..., 1] *= stretch
        c, s = np.cos(angle), np.sin(angle)
        pts = pts @ np.array([[c, -s], [s, c]], np.float32).T * scale + shift

    shoulders = pts[:, POSE + np.array([L_SHOULDER, R_SHOULDER])]   # T x 2 x 2
    ok = np.isfinite(shoulders).all(axis=(1, 2))
    if not ok.any():
        return None
    shoulders = _fill(shoulders, ok)
    center = shoulders.mean(axis=1)                                  # T x 2
    width = np.median(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    pts = (pts - center[:, None]) / max(width, 1e-6)

    seen = np.isfinite(pts[:, [0, 21], 0])                          # T x 2
    if seen[:, 0].sum() > seen[:, 1].sum():      # hand 0 is the main one: mirror
        pts[..., 0] = -pts[..., 0]
        pts = np.concatenate([pts[:, 21:42], pts[:, 0:21], pts[:, 42:]], axis=1)
        pose = pts[:, 42:].copy()
        for a, b in ((11, 12), (13, 14), (15, 16)):                  # swap left/right body
            pose[:, [a, b]] = pose[:, [b, a]]
        pts[:, 42:] = pose
        seen = seen[:, ::-1]
    return pts, seen


def frame_features(pts, seen):
    """T x 75 x 2 -> T x FRAME_DIM (missing hands are zeros with flag 0)."""
    T = len(pts)
    out = []
    for start, col in ((21, 1), (0, 0)):        # main hand first (points 21-41, see above)
        hand = np.nan_to_num(pts[:, start:start + 21])
        ok = seen[:, col]
        wrist = np.where(ok[:, None], hand[:, 0], 0.0)
        rel = hand - hand[:, :1]
        size = np.linalg.norm(rel, axis=2).max(axis=1)
        shape = np.where(ok[:, None, None], rel / np.maximum(size, 1e-6)[:, None, None], 0.0)
        out += [wrist, shape.reshape(T, 42), ok[:, None].astype(np.float32)]
    upper = np.nan_to_num(pts[:, POSE + np.array(UPPER)], nan=0.0).reshape(T, -1)
    out.append(upper)
    return np.nan_to_num(np.concatenate(out, axis=1).astype(np.float32))


def resample(x, steps=STEPS):
    """T x D -> steps x D by linear interpolation in time."""
    T = len(x)
    if T == 1:
        return np.repeat(x, steps, axis=0)
    pos = np.linspace(0, T - 1, steps)
    i = np.floor(pos).astype(int)
    j = np.minimum(i + 1, T - 1)
    w = (pos - i)[:, None]
    return (1 - w) * x[i] + w * x[j]


def sequence(clip, aspect=ISLR_ASPECT, augment=None):
    """clip -> STEPS x FRAME_DIM, or None if there are no shoulders or no hands."""
    norm = normalize_clip(clip, aspect, augment)
    if norm is None:
        return None
    pts, seen = norm
    keep = seen.any(axis=1)                   # only frames where a hand is seen
    if keep.sum() < 2:
        return None
    first, last = np.argmax(keep), len(keep) - np.argmax(keep[::-1])
    return resample(frame_features(pts[first:last], seen[first:last])).astype(np.float32)


def summary(seq):
    """STEPS x FRAME_DIM -> one vector for SVM / Random Forest: the sequence averaged in
    8 parts of time, plus the mean and the std over the whole clip."""
    parts = seq.reshape(8, -1, seq.shape[1]).mean(axis=1).ravel()
    return np.concatenate([parts, seq.mean(axis=0), seq.std(axis=0)])


def random_augment(rng):
    return (rng.uniform(0.85, 1.2), rng.uniform(-0.15, 0.15), rng.uniform(0.9, 1.1),
            rng.uniform(-0.05, 0.05, 2).astype(np.float32))
