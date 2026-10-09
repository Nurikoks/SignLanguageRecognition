"""Hand landmark extraction with MediaPipe Hands (Tasks API).

Note: new MediaPipe versions (0.10.30+) removed the old `mp.solutions` API,
so this code uses the Tasks API (HandLandmarker) instead.
"""
from itertools import permutations

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 static letters (no J and Z)

# pairs of landmark ids that form the hand skeleton (used for drawing)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),
]
HAND_COLORS = {"Left": (0, 140, 255), "Right": (255, 160, 0)}   # BGR: orange, blue
SIDES = ("Left", "Right")
PALM = [0, 5, 9, 13, 17]   # wrist and finger bases: the palm center moves less noisily


def create_detector(video=False, num_hands=1):
    """IMAGE mode for dataset images, VIDEO mode for the webcam."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        # lower than the default 0.5, so a fast (blurry) hand is followed instead of lost
        min_hand_presence_confidence=0.4 if video else 0.5,
        min_tracking_confidence=0.3 if video else 0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(detector, rgb, timestamp_ms=None, tracker=None):
    """Return {'Left': 21x3 array, 'Right': 21x3 array} in pixel units with the hands found.
    For video pass a HandTracker, so hands do not swap sides while they move."""
    h, w = rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    if timestamp_ms is None:
        result = detector.detect(image)
    else:
        result = detector.detect_for_video(image, timestamp_ms)
    hands = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        # MediaPipe gives x and y divided by width and height. We go back to pixels,
        # otherwise a 640x480 webcam frame and a 200x200 dataset image would not match.
        points = np.array([[p.x * w, p.y * h, p.z * w] for p in landmarks], dtype=np.float32)
        hands.append((points, handedness[0].category_name))
    if tracker is not None:
        return tracker.update(timestamp_ms / 1000, hands)
    # Sometimes both hands get the same label. Then use the position on screen:
    # in the mirrored webcam view the right hand is on the right side.
    if len(hands) == 2 and hands[0][1] == hands[1][1]:
        hands.sort(key=lambda hand: hand[0][0, 0])
        hands = [(hands[0][0], "Left"), (hands[1][0], "Right")]
    return {side: points for points, side in hands}


class HandTracker:
    """Keeps 'Left' and 'Right' on the same hands from frame to frame.

    MediaPipe guesses left or right again in every frame, and while a hand moves the
    guess often flips, so the hands seem to swap places. Here a hand keeps its side
    by following its position. MediaPipe's guess is only used for a hand that just
    appeared, or when it says the other side for about a second without a break."""

    MAX_GAP = 0.5     # seconds a lost hand is remembered
    NEW_COST = 1.5    # a jump longer than this (in hand sizes) looks more like a new hand
    VOTE_STEP = 0.05  # how fast the vote follows MediaPipe's guess
    SWAP_VOTE = -0.6  # vote below this: MediaPipe keeps saying the other side

    def __init__(self):
        self.tracks = {}   # side -> (time, palm center, hand size, vote)

    def _cost(self, points, label, side, leftmost):
        track = self.tracks.get(side)
        if track is None:   # new hand: trust MediaPipe's guess, then the screen position
            position = 0.0 if leftmost is None else 0.5 * ((side == "Left") != leftmost)
            return self.NEW_COST + (label != side) + position
        return np.linalg.norm(points[PALM, :2].mean(axis=0) - track[1]) / track[2]

    def update(self, t, found):
        """found: list of (points, MediaPipe label). Returns {side: points}."""
        self.tracks = {s: tr for s, tr in self.tracks.items() if t - tr[0] <= self.MAX_GAP}
        xs = [points[PALM, 0].mean() for points, _ in found]
        leftmost = [x == min(xs) if len(found) == 2 else None for x in xs]
        # try both ways to give the hands their sides, keep the one with the smallest moves
        option = min(permutations(SIDES, len(found)), key=lambda sides: sum(
            self._cost(points, label, side, left)
            for (points, label), side, left in zip(found, sides, leftmost)))

        hands, votes = {}, {}
        for (points, label), side in zip(found, option):
            agree = 1.0 if label == side else -1.0
            old = self.tracks[side][3] if side in self.tracks else 0.0
            hands[side] = points
            votes[side] = (1 - self.VOTE_STEP) * old + self.VOTE_STEP * agree                 if side in self.tracks else 0.5 * agree
        if hands and all(votes[s] < self.SWAP_VOTE for s in hands):
            other = {"Left": "Right", "Right": "Left"}
            hands = {other[s]: p for s, p in hands.items()}
            votes = {other[s]: -v for s, v in votes.items()}
            self.tracks = {}
        for side, points in hands.items():
            self.tracks[side] = (t, points[PALM, :2].mean(axis=0), max(hand_size(points), 1.0),
                                 votes[side])
        return hands


def detect(detector, rgb, timestamp_ms=None):
    """Return (21x3 array in pixel units, 'Left'/'Right') or (None, None) if no hand."""
    for side, points in detect_hands(detector, rgb, timestamp_ms).items():
        return points, side
    return None, None


def hand_size(points):
    """Largest distance (in x, y) from the wrist to another landmark."""
    return float(np.max(np.linalg.norm(points[:, :2] - points[0, :2], axis=1)))


def normalize(points, side):
    """Turn 21 landmarks into 63 features that do not depend on
    hand position, hand size or which hand is used."""
    pts = points - points[0]              # wrist becomes (0, 0, 0)
    if side == "Left":
        pts[:, 0] = -pts[:, 0]            # mirror, so both hands look the same
    scale = hand_size(pts)
    if scale > 0:
        pts = pts / scale                 # hand size becomes 1
    return pts.flatten()


def draw_hand(frame, points, color=(0, 140, 255)):
    for a, b in HAND_CONNECTIONS:
        pa, pb = points[a, :2].astype(int), points[b, :2].astype(int)
        cv2.line(frame, tuple(pa), tuple(pb), (255, 255, 255), 2)
    for p in points[:, :2].astype(int):
        cv2.circle(frame, tuple(p), 4, color, -1)
