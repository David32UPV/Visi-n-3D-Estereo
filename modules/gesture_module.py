"""Reconocimiento de gestos con MediaPipe para frames estéreo.

Este módulo encapsula la carga del modelo de gestos de MediaPipe y la
interfaz mínima para procesar imágenes BGR y obtener las etiquetas de gestos
detectadas junto con la imagen anotada.
"""

from __future__ import annotations

import os
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np


class GestureRecognizer:
    """Reconoce gestos de la mano usando MediaPipe Gesture Recognizer."""

    def __init__(self, model_path: str = "gesture_recognizer.task"):
        """
        Inicializa el reconocedor de gestos.
        Descarga el modelo si no existe.
        """
        try:
            import mediapipe as mp
            from mediapipe.framework.formats import landmark_pb2
        except ImportError as exc:
            raise ImportError(
                "MediaPipe es necesario para el modo de gestos. Instálalo con: pip install mediapipe"
            ) from exc

        self.mp = mp
        self.landmark_pb2 = landmark_pb2
        self.model_path = model_path
        self._download_model_if_needed()

        print("Inicializando MediaPipe Gesture Recognizer...")
        BaseOptions = self.mp.tasks.BaseOptions
        GestureRecognizerTask = self.mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = self.mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = self.mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )
        self.recognizer = GestureRecognizerTask.create_from_options(options)

        self.mp_drawing = self.mp.solutions.drawing_utils
        self.mp_hands = self.mp.solutions.hands

    def _download_model_if_needed(self):
        """Descarga el modelo oficial de MediaPipe si no existe localmente."""
        if not os.path.exists(self.model_path):
            print("Descargando el modelo de gestos de MediaPipe (solo primera ejecución)...")
            url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Modelo descargado.")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Procesa un frame, detecta manos, dibuja el esqueleto y la hitbox,
        y devuelve la imagen anotada junto con la lista de gestos detectados.

        Args:
            frame: Imagen BGR de la cámara.

        Returns:
            Tupla (frame_anotado, lista_de_gestos).
        """
        result_frame = frame.copy()
        detected_gestures = []

        # MediaPipe espera imágenes RGB
        rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)

        # Inferencia
        recognition_result = self.recognizer.recognize(mp_image)

        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # A) Dibujar el esqueleto
                hand_landmarks_proto = self.landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    self.landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
                ])

                self.mp_drawing.draw_landmarks(
                    result_frame,
                    hand_landmarks_proto,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

                # B) Calcular la hitbox (bounding box)
                h, w, _ = result_frame.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks]
                y_coords = [int(lm.y * h) for lm in hand_landmarks]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(result_frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 255), 2)

                # C) Obtener el gesto y escribir el texto
                gesture_name = recognition_result.gestures[i][0].category_name
                
                if gesture_name != "None":
                    cv2.putText(result_frame, gesture_name, (x_min - 20, y_min - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    detected_gestures.append(gesture_name)

        return result_frame, detected_gestures
