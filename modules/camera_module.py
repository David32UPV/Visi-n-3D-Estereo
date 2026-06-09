"""Wrapper ligero para captura estéreo con ZED 2.

Proporciona una interfaz mínima para abrir la cámara ZED, capturar
frames de las lentes izquierda y derecha y cerrar la cámara.

Notas:
- Este módulo depende del SDK `pyzed` (pyzed.sl). Si no dispone de
  la cámara o del SDK, las llamadas a `open()` fallarán en tiempo de ejecución.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl


class ZEDCamera:
    """Pequeña clase envoltorio para la API de captura estéreo de ZED.

    Uso típico:
        cam = ZEDCamera()
        cam.open()
        while cam.grab():
            left, right = cam.get_frames()
        cam.close()
    """

    def __init__(self, resolution: sl.RESOLUTION = sl.RESOLUTION.HD720, fps: int = 30):
        # Parámetros de captura configurables
        self.resolution = resolution
        self.fps = fps

        # Objeto SDK y parámetros de ejecución
        self.camera = sl.Camera()
        self.runtime = sl.RuntimeParameters()

        # Mats que usarán el SDK para devolver imágenes
        self.left_mat: Optional[sl.Mat] = None
        self.right_mat: Optional[sl.Mat] = None
        self._is_open = False
        self.last_open_status: Optional[sl.ERROR_CODE] = None

        # Inicializar parámetros de apertura de la ZED
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        # No necesitamos depth en este ejemplo (solo imágenes BGR)
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE

    def open(self) -> bool:
        """Abre la cámara ZED con los parámetros configurados.

        Devuelve True si la cámara se abrió correctamente.
        """
        status = self.camera.open(self.init_params)
        self.last_open_status = status
        self._is_open = status == sl.ERROR_CODE.SUCCESS

        if self._is_open:
            # Precrear mats para evitar asignaciones en cada frame
            self.left_mat = sl.Mat()
            self.right_mat = sl.Mat()

        return self._is_open

    def grab(self) -> bool:
        """Captura el siguiente frame estéreo.

        Devuelve True si hay un nuevo frame disponible (grab éxito).
        """
        if not self._is_open:
            return False
        return self.camera.grab(self.runtime) == sl.ERROR_CODE.SUCCESS

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Recupera las imágenes de la lente izquierda y derecha.

        Devuelve una tupla `(frame_left_BGR, frame_right_BGR)` con formatos
        adecuados para procesar con OpenCV.
        """
        if not self._is_open or self.left_mat is None or self.right_mat is None:
            raise RuntimeError("Cámara no abierta. Llama a open() antes de get_frames().")

        # Solicitar al SDK las imágenes de cada vista
        self.camera.retrieve_image(self.left_mat, sl.VIEW.LEFT)
        self.camera.retrieve_image(self.right_mat, sl.VIEW.RIGHT)

        # El SDK devuelve BGRA; convertimos a BGR para compatibilidad con OpenCV
        frame_left = cv2.cvtColor(self.left_mat.get_data(), cv2.COLOR_BGRA2BGR)
        frame_right = cv2.cvtColor(self.right_mat.get_data(), cv2.COLOR_BGRA2BGR)
        return frame_left, frame_right

    def close(self) -> None:
        """Cierra la cámara y libera recursos del SDK."""
        if self._is_open:
            self.camera.close()
            self._is_open = False
