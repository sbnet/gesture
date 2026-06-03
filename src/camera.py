"""Thin abstraction over OpenCV VideoCapture."""

from __future__ import annotations

import sys

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def _backend() -> int:
    # V4L2 avoids OpenCV 4.13 overload-resolution failures on Linux/WSL2.
    # DirectShow is the reliable choice on Windows.
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


class Camera:
    """Opens a webcam and provides frames one at a time."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(int(index), _backend())
        if not self._cap.isOpened():
            raise CameraError(
                f"Cannot open camera {index}. "
                "Check that the device is connected and not used by another application."
            )
        # MJPEG avoids V4L2 select() timeouts on USB cameras in WSL2.
        # Must be set before resolution so the driver reconfigures in MJPEG mode.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if capture failed."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
