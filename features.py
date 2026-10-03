"""Hand landmark extraction with MediaPipe Hands (Tasks API).

Note: new MediaPipe versions (0.10.30+) removed the old `mp.solutions` API,
so this code uses the Tasks API (HandLandmarker) instead.
"""
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


def create_detector(video=False):
    """IMAGE mode for dataset images, VIDEO mode for the webcam."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(detector, rgb, timestamp_ms=None):
    """Return (21x3 array in pixel units, 'Left'/'Right') or (None, None) if no hand."""
    h, w = rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    if timestamp_ms is None:
        result = detector.detect(image)
    else:
        result = detector.detect_for_video(image, timestamp_ms)
    if not result.hand_landmarks:
        return None, None
    # MediaPipe gives x and y divided by width and height. We go back to pixels,
    # otherwise a 640x480 webcam frame and a 200x200 dataset image would not match.
    points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[0]],
                      dtype=np.float32)
    side = result.handedness[0][0].category_name
    return points, side


def normalize(points, side):
    """Turn 21 landmarks into 63 features that do not depend on
    hand position, hand size or which hand is used."""
    pts = points - points[0]              # wrist becomes (0, 0, 0)
    if side == "Left":
        pts[:, 0] = -pts[:, 0]            # mirror, so both hands look the same
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts = pts / scale                 # hand size becomes 1
    return pts.flatten()
