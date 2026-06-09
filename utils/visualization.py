"""Funciones sencillas de visualización usadas por la demo estéreo.

Este módulo contiene ayudas pequeñas para dibujar marcadores y etiquetas
en imágenes OpenCV, pensado para uso rápido durante depuración.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def draw_points_and_text(
    frame: np.ndarray,
    points: Iterable[Tuple[int, int]],
    text: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Dibuja puntos y un texto opcional en el `frame`.

    Args:
        frame: Imagen BGR sobre la que dibujar.
        points: Iterable de tuplas (x,y) en coordenadas de píxel.
        text: Texto opcional que se dibuja en la esquina superior izquierda.
        color: Color BGR para los puntos y el texto.

    Devuelve la imagen anotada (no modifica el `frame` original).
    """
    out = frame.copy()

    # Dibujar cada punto como un círculo relleno
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 6, color, -1)

    # Si se proporciona texto, dibujarlo en la esquina superior izquierda
    if text:
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return out
