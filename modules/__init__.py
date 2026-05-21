"""Core modules for v2 stereo bin-picking pipeline."""

from .camera_module import ZEDCamera
from .stereo_module import StereoTriangulator
from .gesture_module import GestureRecognizer

__all__ = [
    "ZEDCamera",
    "StereoTriangulator",
    "GestureRecognizer",
]
