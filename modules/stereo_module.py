"""
Módulo de triangulación estéreo y utilidades de disparidad para ZED2.

Este archivo contiene la clase `StereoTriangulator` que encapsula:
- Carga de parámetros de calibración desde un archivo NPZ.
- Cálculo de rectificación estéreo (matrices R1,R2,P1,P2 y Q).
- Mapas de remapeo para rectificar rápidamente pares de imágenes.
- Triangulación esparsa de puntos coincidentes (cv2.triangulatePoints).
- Cálculo de disparidad densa con StereoSGBM y reproyección a 3D.

Notas importantes:
- Las imágenes deben estar en el mismo tamaño que se usó para la calibración
  (o indicar `image_size` correcto al inicializar).
- La matriz `Q` producida por `stereoRectify` sirve para convertir mapas de
  disparidad en coordenadas 3D mediante `cv2.reprojectImageTo3D`.

Las unidades devueltas por la triangulación están en las mismas unidades que
la traslación `T` de la calibración (normalmente milímetros si la calibración
se realizó con esa unidad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class StereoCalibration:
    """Contenedor simple para los parámetros de calibración estéreo.

    Atributos:
        Kl: Matriz intrínseca de la cámara izquierda (3x3).
        Dl: Coeficientes de distorsión de la cámara izquierda.
        Kr: Matriz intrínseca de la cámara derecha (3x3).
        Dr: Coeficientes de distorsión de la cámara derecha.
        R: Rotación de la cámara izquierda a la derecha (3x3).
        T: Traslación de la cámara izquierda a la derecha (3x1).
    """

    Kl: np.ndarray
    Dl: np.ndarray
    Kr: np.ndarray
    Dr: np.ndarray
    R: np.ndarray
    T: np.ndarray


class StereoTriangulator:
    """Clase que agrupa operaciones habituales de visión estéreo.

    Funcionalidades principales:
    - Calcular y almacenar las matrices de rectificación (R1,R2,P1,P2,Q).
    - Construir mapas de remapeo (mapx,mapy) para `cv2.remap`.
    - Triangular puntos individuales a coordenadas 3D.
    - Calcular mapa de disparidad denso y reproyectarlo a 3D.

    Parámetros:
        calib_path: Ruta al archivo NPZ con Kl,Dl,Kr,Dr,R,T.
        image_size: Tupla (ancho, alto) usada para la rectificación.
    """

    def __init__(self, calib_path: str, image_size: Tuple[int, int] = (1280, 720)):
        # Ruta del fichero de calibración y tamaño de imagen esperado.
        self.calib_path = calib_path
        self.image_size = image_size  # (width, height)

        # Cargar parámetros de calibración desde NPZ.
        self.calibration = self._load_calibration(calib_path)

        # Matrices que se calcularán en la rectificación
        self.R1: np.ndarray
        self.R2: np.ndarray
        self.P1: np.ndarray
        self.P2: np.ndarray
        self.Q: np.ndarray
        # Ejecutar stereoRectify para obtener estas matrices.
        self._compute_rectification()

        # Mapas para remap (usados por cv2.remap) — se construyen a partir de P1/P2.
        self.map1x: np.ndarray
        self.map1y: np.ndarray
        self.map2x: np.ndarray
        self.map2y: np.ndarray
        self._build_rectification_maps()

        # Configuración por defecto para el matcher denso StereoSGBM.
        # Estos parámetros se pueden ajustar según escena/ruido.
        block_size = 5
        num_disparities = 16 * 10  # debe ser múltiplo de 16
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    @staticmethod
    def _load_calibration(calib_path: str) -> StereoCalibration:
        """Carga y valida el contenido mínimo del NPZ de calibración.

        El NPZ debe contener las claves: 'Kl','Dl','Kr','Dr','R','T'.
        Lanza ValueError si falta alguna clave.
        """
        data = np.load(calib_path)
        required = ["Kl", "Dl", "Kr", "Dr", "R", "T"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Faltan claves en {calib_path}: {missing}")

        return StereoCalibration(
            Kl=data["Kl"],
            Dl=data["Dl"],
            Kr=data["Kr"],
            Dr=data["Dr"],
            R=data["R"],
            T=data["T"],
        )

    def _compute_rectification(self) -> None:
        """Calcula las matrices de rectificación con `cv2.stereoRectify`.

        Usamos `CALIB_ZERO_DISPARITY` para intentar que las imágenes rectificadas
        tengan un eje de proyección común (y minimizar el sesgo horizontal).

        Las rectas epipolares resultantes deberían ser horizontales, lo que facilita la búsqueda
        """
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.calibration.Kl,
            self.calibration.Dl,
            self.calibration.Kr,
            self.calibration.Dr,
            self.image_size,
            self.calibration.R,
            self.calibration.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

    def _build_rectification_maps(self) -> None:
        """Construye los mapas de remapeo para `cv2.remap`.

        Estos mapas transforman una imagen original (distorsionada) a su
        versión rectificada, lista para buscar correspondencias fila a fila.
        """
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.calibration.Kl,
            self.calibration.Dl,
            self.R1,
            self.P1,
            self.image_size,
            cv2.CV_32FC1,
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.calibration.Kr,
            self.calibration.Dr,
            self.R2,
            self.P2,
            self.image_size,
            cv2.CV_32FC1,
        )

    def rectify(self, frame_left: np.ndarray, frame_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rectifica un par de imágenes estéreo.

        Args:
            frame_left: Imagen BGR original de la cámara izquierda.
            frame_right: Imagen BGR original de la cámara derecha.

        Devuelve:
            rect_left, rect_right: imágenes rectificadas listas para comparación.
        """
        rect_left = cv2.remap(frame_left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def triangulate_points(
        self,
        pts_left: Sequence[Sequence[float]],
        pts_right: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Triangula pares de puntos correspondientes en imágenes rectificadas.

        Notas sobre entradas/salidas:
        - `pts_left` y `pts_right` deben ser listas/arrays Nx2 con coordenadas
          de píxel (u,v) en las imágenes rectificadas.
        - Se devuelve un array Nx3 con coordenadas 3D (X,Y,Z) en las unidades de T.

        El método usa `cv2.triangulatePoints` que acepta coordenadas en forma 2xN.
        """
        left = np.asarray(pts_left, dtype=np.float64).reshape(-1, 2)
        right = np.asarray(pts_right, dtype=np.float64).reshape(-1, 2)

        if left.shape[0] != right.shape[0]:
            raise ValueError("pts_left y pts_right deben tener el mismo número de puntos")
        if left.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)

        # cv2.triangulatePoints requiere arrays 2xN
        left_2xN = left.T
        right_2xN = right.T

        # points_4d es 4xN en coordenadas homogéneas
        pts_4d = cv2.triangulatePoints(self.P1, self.P2, left_2xN, right_2xN)
        # Normalizar homogéneas -> 3D Euclídeas
        pts_3d = (pts_4d[:3] / pts_4d[3]).T
        return pts_3d

    def compute_disparity(self, rect_left: np.ndarray, rect_right: np.ndarray) -> np.ndarray:
        """Calcula el mapa de disparidad (float) entre dos imágenes rectificadas.

        Devuelve la disparidad en píxeles (float), donde valores mayores indican
        objetos más cercanos (dependiendo de la configuración del matcher).
        """
        gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY) if rect_left.ndim == 3 else rect_left
        gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY) if rect_right.ndim == 3 else rect_right

        # StereoSGBM devuelve 16*disparity como entero; normalizamos a float
        disparity = self.sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity

    def disparity_to_3d(self, disparity: np.ndarray) -> np.ndarray:
        """Reproyecta un mapa de disparidad a coordenadas 3D usando la matriz Q.

        El resultado es una imagen HxWx3 con coordenadas (X,Y,Z) para cada píxel.
        Valores inválidos (sin disparidad) pueden aparecer como grandes o NaN.
        """
        return cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)

    def _sample_disparity_window(
        self,
        disparity: np.ndarray,
        u: float,
        v: float,
        window: int = 5,
    ) -> Optional[float]:
        """Disparidad robusta en una ventana alrededor de (u,v).

        En lugar de leer un único píxel (que en SGBM suele caer en un hueco o
        en ruido), tomamos la mediana de las disparidades válidas (>0 y finitas)
        en un cuadrado de lado `2*window+1`. Esto estabiliza tanto la profundidad
        como la proyección del centroide a la vista derecha.

        Devuelve `None` si no hay ningún valor válido en la ventana.
        """
        h, w = disparity.shape[:2]
        cu = int(round(float(u)))
        cv_ = int(round(float(v)))
        if cu < 0 or cv_ < 0 or cu >= w or cv_ >= h:
            return None

        u0, u1 = max(0, cu - window), min(w, cu + window + 1)
        v0, v1 = max(0, cv_ - window), min(h, cv_ + window + 1)
        patch = disparity[v0:v1, u0:u1]

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def _reproject_point(
        self,
        u: float,
        v: float,
        d: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Convierte un único punto (u,v,disparidad) a 3D usando la matriz Q.

        Equivale a lo que hace `cv2.reprojectImageTo3D` por píxel, pero sin
        depender de que el píxel exacto tenga disparidad válida en el mapa.
        """
        homog = self.Q @ np.array([float(u), float(v), float(d), 1.0], dtype=np.float64)
        if abs(homog[3]) < 1e-9:
            return None

        point = homog[:3] / homog[3]
        if not np.all(np.isfinite(point)):
            return None

        return float(point[0]), float(point[1]), float(point[2])

    def get_3d_from_pixel(
        self,
        disparity: np.ndarray,
        u: int,
        v: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Obtiene la coordenada 3D (X,Y,Z) de un píxel del mapa de disparidad.

        Usa una disparidad robusta (mediana de una ventana) y reproyecta ese
        único punto con la matriz Q. Devuelve `None` si no hay disparidad válida
        en la vecindad.
        """
        d = self._sample_disparity_window(disparity, u, v)
        if d is None:
            return None

        return self._reproject_point(u, v, d)

    def get_3d_from_bbox_center(
        self,
        disparity: np.ndarray,
        bbox: Sequence[float],
    ) -> Optional[Tuple[float, float, float]]:
        """Obtiene la coordenada 3D del centro de un bbox usando la disparidad.

        El bbox debe venir en formato (x1, y1, x2, y2) sobre la imagen rectificada.
        El punto de muestreo se toma como el centro geométrico de la caja.
        """
        if len(bbox) < 4:
            raise ValueError("El bbox debe contener al menos 4 valores: x1, y1, x2, y2")

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return self.get_3d_from_centroid(disparity, center)

    def get_3d_from_centroid(
        self,
        disparity: np.ndarray,
        centroid: Sequence[float],
    ) -> Optional[Tuple[float, float, float]]:
        """Obtiene la coordenada 3D de un centroide (u,v) en la imagen rectificada izquierda."""
        if len(centroid) < 2:
            raise ValueError("El centroide debe contener al menos 2 valores: u, v")

        center_u = int(round(float(centroid[0])))
        center_v = int(round(float(centroid[1])))
        return self.get_3d_from_pixel(disparity, center_u, center_v)

    def get_disparity_at_bbox_center(
        self,
        disparity: np.ndarray,
        bbox: Sequence[float],
    ) -> Optional[float]:
        """Devuelve la disparidad exacta del centro geométrico de un bbox."""
        if len(bbox) < 4:
            raise ValueError("El bbox debe contener al menos 4 valores: x1, y1, x2, y2")

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return self.get_disparity_at_centroid(disparity, center)

    def get_disparity_at_centroid(
        self,
        disparity: np.ndarray,
        centroid: Sequence[float],
    ) -> Optional[float]:
        """Devuelve la disparidad robusta de un centroide (u,v) en la imagen rectificada izquierda.

        Toma la mediana de una ventana alrededor del centroide para evitar el
        ruido y los huecos del mapa SGBM en el píxel exacto.
        """
        if len(centroid) < 2:
            raise ValueError("El centroide debe contener al menos 2 valores: u, v")

        return self._sample_disparity_window(disparity, centroid[0], centroid[1])

    def get_right_center_from_bbox_center(
        self,
        disparity: np.ndarray,
        bbox: Sequence[float],
    ) -> Optional[Tuple[int, int]]:
        """Projeta el centro de un bbox de la izquierda a la vista derecha rectificada."""
        if len(bbox) < 4:
            raise ValueError("El bbox debe contener al menos 4 valores: x1, y1, x2, y2")

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return self.get_right_center_from_centroid(disparity, center)

    def get_right_center_from_centroid(
        self,
        disparity: np.ndarray,
        centroid: Sequence[float],
    ) -> Optional[Tuple[int, int]]:
        """Projeta un centroide de la izquierda a la vista derecha rectificada usando disparidad."""
        if len(centroid) < 2:
            raise ValueError("El centroide debe contener al menos 2 valores: u, v")

        center_u = int(round(float(centroid[0])))
        center_v = int(round(float(centroid[1])))

        d = self.get_disparity_at_centroid(disparity, centroid)
        if d is None:
            return None

        right_u = int(round(center_u - d))
        return right_u, center_v

    @staticmethod
    def draw_epilines(rect_left: np.ndarray, rect_right: np.ndarray, step: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """Dibuja líneas epipolares horizontales para comprobar la rectificación.

        Esto es útil visualmente: tras rectificar, puntos idénticos deberían
        aparecer en la misma fila (misma coordenada Y) en ambas imágenes.
        """
        out_left = rect_left.copy()
        out_right = rect_right.copy()

        height = rect_left.shape[0]
        for y in range(0, height, step):
            cv2.line(out_left, (0, y), (out_left.shape[1], y), (0, 255, 255), 1)
            cv2.line(out_right, (0, y), (out_right.shape[1], y), (0, 255, 255), 1)

        return out_left, out_right
