"""Open the webcam with the wanted size and frame rate, and measure the real frame rate.

On Windows the default OpenCV backend often ignores size and fps settings, DirectShow
follows them. The camera picks the closest mode it supports, so the real mode is
printed and the real fps is shown in the window.
"""
import sys
import time
from collections import deque

import cv2


def add_camera_args(parser):
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60, help="the camera gives the most it can")
    parser.add_argument("--exposure", type=float, default=None,
                        help="manual exposure, e.g. -6 (Windows: 2^-6 s). In a dark room auto "
                             "exposure halves the fps; a manual short one keeps it, but darker")


def open_camera(args):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    # compressed frames: many cameras need them for HD at a high fps (if they support it)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual for DirectShow
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h} (asked {args.width}x{args.height} at {args.fps} fps). "
          f"The real fps is shown in the window")
    return cap


def open_window(name, frame_width, frame_height):
    """A window you can resize with the mouse; starts bigger for small frames."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    k = max(1.0, 720 / frame_height)
    cv2.resizeWindow(name, int(frame_width * k), int(frame_height * k))


class FpsCounter:
    def __init__(self):
        self.times = deque(maxlen=30)

    def tick(self):
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])
