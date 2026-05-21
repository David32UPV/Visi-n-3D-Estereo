"""Captura de imágenes ZED para crear un dataset pre-etiquetado.

Este modulo guarda una captura puntual de la ZED en disco para que luego
puedas etiquetarla en Roboflow y reentrenar el detector.

Por defecto guarda solo la lente izquierda en `images_pre_labeled/`, que es
suficiente para entrenar un detector 2D. Si alguna vez quieres conservar las
dos vistas, activa `save_both_views=True` y se guardaran en subcarpetas
`left/` y `right/`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


class PrelabelCaptureModule:
    """Guarda capturas de la ZED para etiquetado posterior."""

    def __init__(self, output_root: str | Path = "images_pre_labeled", save_both_views: bool = False) -> None:
        self.output_root = Path(output_root)
        self.save_both_views = save_both_views

        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.save_both_views:
            (self.output_root / "left").mkdir(parents=True, exist_ok=True)
            (self.output_root / "right").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _write_jpg(path: Path, frame: np.ndarray) -> None:
        ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"No se ha podido guardar la imagen en: {path}")

    def capture(self, frame_left: np.ndarray, frame_right: np.ndarray) -> Dict[str, Path]:
        """Guarda una captura de la lente izquierda o de ambas, segun la configuracion."""
        stamp = self._timestamp()
        saved_paths: Dict[str, Path] = {}

        if self.save_both_views:
            left_path = self.output_root / "left" / f"zed_left_{stamp}.jpg"
            right_path = self.output_root / "right" / f"zed_right_{stamp}.jpg"
            self._write_jpg(left_path, frame_left)
            self._write_jpg(right_path, frame_right)
            saved_paths["left"] = left_path
            saved_paths["right"] = right_path
        else:
            left_path = self.output_root / f"zed_left_{stamp}.jpg"
            self._write_jpg(left_path, frame_left)
            saved_paths["left"] = left_path

        return saved_paths


__all__ = ["PrelabelCaptureModule"]
