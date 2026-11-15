"""Word recognition for the webcam demo (model from train_islr.py).

Every frame we build the same 75 x 3 array as in the Kaggle ISLR clips: two hands and
the pose in image coordinates (x / width, y / height). A word is recorded from the
moment a hand appears until the hands are lowered (no hand for a short time) or stay
still after moving. Then the whole recording is classified once.
"""
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from word_features import sequence

POSE_MODEL = "pose_landmarker_lite.task"
WORDS_NET = "words_model.pt"

END_NO_HANDS = 0.35    # seconds without hands: the word is over
END_STILL = 0.6        # seconds of still hands after moving: the word is over
STILL_SPEED = 0.6      # wrist speed below this (shoulder widths per second) is "still"
MIN_SECONDS = 0.4      # shorter recordings are ignored
MAX_SECONDS = 4.0      # longer recordings are dropped (probably not a word)
MIN_PROB = 0.45        # the best word must be at least this sure to be written


def create_pose_detector():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    return vision.PoseLandmarker.create_from_options(options)


def frame_array(pose_detector, rgb, timestamp_ms, hands):
    """hands: {'Left'/'Right': 21x3 pixels} from features.detect_hands.
    Returns 75 x 3 (NaN where missing) in image coordinates, or None without a pose."""
    h, w = rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = pose_detector.detect_for_video(image, timestamp_ms)
    if not result.pose_landmarks:
        return None
    out = np.full((75, 3), np.nan, np.float32)
    out[42:] = [[p.x, p.y, p.z] for p in result.pose_landmarks[0]]
    wrists = out[42 + np.array([15, 16]), :2]          # pose left and right wrist
    slots = {}
    for points in hands.values():
        xy = points[:, :2] / [w, h]
        # like MediaPipe Holistic in the dataset: a hand belongs to the nearest pose wrist
        slot = int(np.argmin(np.linalg.norm(wrists - xy[0], axis=1)))
        if slot in slots:      # both near the same wrist: give it to the other slot
            slot = 1 - slot
        slots[slot] = True
        out[slot * 21:(slot + 1) * 21, :2] = xy
        out[slot * 21:(slot + 1) * 21, 2] = points[:, 2] / w
    return out


class WordModel:
    def __init__(self, path=WORDS_NET):
        from train_islr import NETS
        data = torch.load(path, map_location="cpu")
        self.words = data["words"]
        self.model = NETS[data["model"]](len(self.words))
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()

    def top(self, clip, aspect, k=3):
        x = sequence(clip, aspect=aspect)
        if x is None:
            return []
        with torch.inference_mode():
            p = torch.softmax(self.model(torch.tensor(x)[None]), dim=1)[0].numpy()
        best = np.argsort(p)[::-1][:k]
        return [(self.words[i], float(p[i])) for i in best]


class WordRecorder:
    """Collects frames while a word is made and says when the word is over."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames, self.times = [], []
        self.last_hand, self.moved, self.still_since = None, False, None

    @property
    def recording(self):
        return bool(self.frames)

    def add(self, t, frame):
        """frame: 75 x 3 or None. Returns the finished clip (frames x 75 x 3) or None."""
        has_hand = frame is not None and np.isfinite(frame[[0, 21], 0]).any()
        if has_hand:
            if not self.frames:
                self.reset()
            self.frames.append(frame)
            self.times.append(t)
            self.last_hand = t
            self._update_motion()
        elif self.frames:
            if frame is not None:
                self.frames.append(frame)
                self.times.append(t)
            if t - self.last_hand >= END_NO_HANDS:
                return self._finish()
            return None
        if self.frames and self.moved and self.still_since and t - self.still_since >= END_STILL:
            return self._finish()
        if self.frames and self.times[-1] - self.times[0] > MAX_SECONDS:
            self.reset()
        return None

    def _update_motion(self):
        if len(self.frames) < 3:
            return
        a, b = self.frames[-3], self.frames[-1]
        shoulders = np.linalg.norm(b[42 + 11, :2] - b[42 + 12, :2]) or 1.0
        dt = max(self.times[-1] - self.times[-3], 1e-3)
        speeds = [np.linalg.norm(b[s, :2] - a[s, :2]) / shoulders / dt
                  for s in (0, 21) if np.isfinite(a[s, 0]) and np.isfinite(b[s, 0])]
        speed = max(speeds) if speeds else 0.0
        if speed >= STILL_SPEED:
            self.moved, self.still_since = True, None
        elif self.still_since is None:
            self.still_since = self.times[-1]

    def _finish(self):
        frames, times = self.frames, self.times
        if self.still_since is not None and self.moved:      # cut the still end
            keep = [i for i, t in enumerate(times) if t <= self.still_since + 0.15]
            frames = [frames[i] for i in keep] or frames
            times = [times[i] for i in keep] or times
        self.reset()
        if times[-1] - times[0] < MIN_SECONDS:
            return None
        return np.stack(frames)

