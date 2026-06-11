"""Reconocimiento de gestos con MediaPipe para frames estéreo.

Usa la Tasks API de mediapipe 0.10.x (única disponible en esa versión).
Los landmarks se dibujan manualmente con OpenCV, sin depender de
mp.solutions ni de landmark_pb2.
"""

from __future__ import annotations

import os
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np

_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
])

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)


class GestureRecognizer:
    """Reconoce gestos de la mano usando la Tasks API de mediapipe 0.10.x."""

    def __init__(self, model_path: str = "gesture_recognizer.task") -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe es necesario para el modo de gestos. Instálalo con: pip install mediapipe"
            ) from exc

        self.model_path = model_path
        self._download_model_if_needed()

        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerTask = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
        )
        self._recognizer = GestureRecognizerTask.create_from_options(options)
        self._mp = mp

    def _download_model_if_needed(self) -> None:
        if not os.path.exists(self.model_path):
            print("Descargando modelo de gestos de MediaPipe (solo primera ejecución)...")
            urllib.request.urlretrieve(_MODEL_URL, self.model_path)
            print("Modelo descargado.")

    @staticmethod
    def _draw_hand(frame: np.ndarray, landmarks) -> None:
        """Dibuja esqueleto y bounding box de una mano directamente con OpenCV."""
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (255, 0, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)

        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cv2.rectangle(
            frame,
            (max(0, min(xs) - 20), max(0, min(ys) - 20)),
            (min(w, max(xs) + 20), min(h, max(ys) + 20)),
            (0, 255, 255), 2,
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Procesa un frame BGR y devuelve la imagen anotada y los gestos detectados."""
        result_frame = frame.copy()
        detected_gestures: List[str] = []

        rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._recognizer.recognize(mp_image)

        if result.hand_landmarks:
            h, w, _ = result_frame.shape
            for i, landmarks in enumerate(result.hand_landmarks):
                self._draw_hand(result_frame, landmarks)

                gesture_name = (
                    result.gestures[i][0].category_name
                    if result.gestures and i < len(result.gestures)
                    else "Unknown"
                )
                if gesture_name and gesture_name != "None":
                    detected_gestures.append(gesture_name)
                    xs = [int(lm.x * w) for lm in landmarks]
                    ys = [int(lm.y * h) for lm in landmarks]
                    cv2.putText(
                        result_frame, gesture_name,
                        (max(0, min(xs) - 20), max(30, min(ys) - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA,
                    )

        return result_frame, detected_gestures
