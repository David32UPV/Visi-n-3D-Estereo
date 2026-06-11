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

# Articulaciones (MCP, PIP, DIP, TIP) de cada dedo según la convención de
# MediaPipe Hands. El pulgar usa (CMC, MCP, IP, TIP).
_FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Un dedo se considera extendido si su articulación intermedia está casi recta.
# Umbral en grados (más alto = más estricto). Invariante a la orientación.
_EXTENDED_ANGLE_DEG = 150.0


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

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Ángulo (grados) en el vértice `b` formado por los segmentos b->a y b->c."""
        ba = a - b
        bc = c - b
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cosang = float(np.dot(ba, bc) / denom)
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    @classmethod
    def _fingers_extended(cls, pts: np.ndarray) -> dict:
        """Para cada dedo, True si está extendido (recto), independiente de la orientación.

        Se mide el ángulo de la articulación intermedia: si el dedo está doblado
        el ángulo es pequeño; si está estirado se acerca a 180°. Esto vale igual
        con la mano apuntando arriba, abajo, izquierda o derecha.
        """
        state = {}
        for name in ("index", "middle", "ring", "pinky"):
            mcp, pip, _dip, tip = _FINGER_JOINTS[name]
            state[name] = cls._angle(pts[mcp], pts[pip], pts[tip]) > _EXTENDED_ANGLE_DEG

        # El pulgar mide la rectitud en su articulación IP.
        _cmc, t_mcp, t_ip, t_tip = _FINGER_JOINTS["thumb"]
        state["thumb"] = cls._angle(pts[t_mcp], pts[t_ip], pts[t_tip]) > _EXTENDED_ANGLE_DEG
        return state

    def _classify_gesture(self, landmarks, w: int, h: int, fallback: str = "Unknown") -> str:
        """Clasifica el gesto a partir del patrón de dedos extendidos/doblados.

        Lógica geométrica (invariante a la orientación de la mano):
        - Palma abierta: los 4 dedos (sin pulgar) extendidos, apunten donde apunten.
        - Puño cerrado: los 4 dedos doblados y el pulgar no apunta claramente arriba/abajo.
        - Pulgar arriba/abajo: 4 dedos doblados y pulgar extendido en vertical.
        - Otros patrones parciales: Pointing_Up, Victory, ILoveYou.
        """
        # Trabajamos en píxeles para que los ángulos no se distorsionen por el 16:9.
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        state = self._fingers_extended(pts)

        four = [state["index"], state["middle"], state["ring"], state["pinky"]]
        n_extended = sum(four)

        # Palma abierta: los cuatro dedos estirados, en cualquier dirección.
        if n_extended == 4:
            return "Open_Palm"

        # Mano cerrada: ningún dedo estirado -> puño o pulgar arriba/abajo.
        if n_extended == 0:
            if state["thumb"]:
                # Dirección del pulgar (punta - base). En imagen, y crece hacia abajo.
                thumb_vec = pts[4] - pts[2]
                if abs(thumb_vec[1]) > abs(thumb_vec[0]):  # predominantemente vertical
                    return "Thumb_Up" if thumb_vec[1] < 0 else "Thumb_Down"
            return "Closed_Fist"

        # Patrones parciales habituales.
        if n_extended == 1 and state["index"]:
            return "Pointing_Up"
        if n_extended == 2 and state["index"] and state["middle"]:
            return "Victory"
        if n_extended == 2 and state["index"] and state["pinky"] and state["thumb"]:
            return "ILoveYou"

        return fallback

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detecta manos en un frame BGR SIN dibujar nada.

        Devuelve una lista de manos, cada una con:
        - `landmarks`: landmarks normalizados de MediaPipe (para dibujar luego),
        - `bbox`: caja envolvente (x1, y1, x2, y2) en píxeles,
        - `gesture`: nombre del gesto reconocido.

        Separar la detección del dibujo permite usar la `bbox` de la mano como
        zona de exclusión para que YOLO no detecte cajas falsas sobre la mano.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize(mp_image)

        hands: List[dict] = []
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                xs = [int(lm.x * w) for lm in landmarks]
                ys = [int(lm.y * h) for lm in landmarks]
                bbox = (
                    max(0, min(xs) - 20),
                    max(0, min(ys) - 20),
                    min(w, max(xs) + 20),
                    min(h, max(ys) + 20),
                )
                # Gesto de MediaPipe como red de seguridad (fallback).
                mp_gesture = (
                    result.gestures[i][0].category_name
                    if result.gestures and i < len(result.gestures)
                    else "Unknown"
                )
                # Clasificación geométrica propia (invariante a la orientación).
                gesture_name = self._classify_gesture(landmarks, w, h, fallback=mp_gesture)
                hands.append({"landmarks": landmarks, "bbox": bbox, "gesture": gesture_name})

        return hands

    def draw(self, frame: np.ndarray, hands: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """Dibuja el esqueleto, la hitbox y el nombre del gesto de cada mano detectada."""
        detected_gestures: List[str] = []
        for hand in hands:
            self._draw_hand(frame, hand["landmarks"])

            gesture_name = hand.get("gesture", "Unknown")
            if gesture_name and gesture_name not in ("None", "Unknown"):
                detected_gestures.append(gesture_name)
                x1, y1, _, _ = hand["bbox"]
                cv2.putText(
                    frame, gesture_name,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA,
                )

        return frame, detected_gestures

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Detecta y dibuja gestos en un frame BGR (detección + dibujo en un paso)."""
        result_frame = frame.copy()
        hands = self.detect(result_frame)
        return self.draw(result_frame, hands)
