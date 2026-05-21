"""Aplicación interactiva para validación de rectificación, triangulación y gestos.

Modos disponibles:
1 - Comprobación de rectificación (dibujar líneas epipolares horizontales)
2 - Triangulación manual por click (click izquierda -> click derecha)
3 - Mapa de disparidad en tiempo real (SGBM)
4 - Reconocimiento de gestos (MediaPipe)

Controles:
- Teclas 1/2/3/4: cambiar modo
- q: salir
- r: limpiar puntos seleccionados (modo 2)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from modules.camera_module import ZEDCamera
from modules.gesture_module import GestureRecognizer
from modules.stereo_module import StereoTriangulator


class StereoInteractiveApp:
    """Aplicación interactiva con varios modos para probar el pipeline estéreo.

    Atributos internos relevantes:
        - `camera`: wrapper de la ZED.
        - `stereo`: instancia de `StereoTriangulator` cargada con calibración.
        - `gesture`: inicializada bajo demanda en modo gestos.
        - `left_click`, `right_click`: coordenadas seleccionadas para triangulación.
    """

    MODE_RAW = 0
    MODE_RECTIFICATION = 1
    MODE_TRIANGULATION = 2
    MODE_DISPARITY = 3
    MODE_GESTURES = 4

    def __init__(self) -> None:
        base_dir = os.path.dirname(__file__)
        calib_path = os.path.join(base_dir, "calibration", "stereo_calib.npz")

        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Fichero de calibración no encontrado: {calib_path}")

        # Inicializaciones principales
        self.camera = ZEDCamera()
        self.stereo = StereoTriangulator(calib_path=calib_path, image_size=(1280, 720))
        self.gesture: Optional[GestureRecognizer] = None

        # Estado del modo y ventanas
        self.mode = self.MODE_RAW
        self.window_left = "ZED2 Left"
        self.window_right = "ZED2 Right"
        self.window_aux = "ZED2 Disparity"

        # Variables para triangulación manual
        self.left_click: Optional[Tuple[int, int]] = None
        self.right_click: Optional[Tuple[int, int]] = None
        self.last_3d: Optional[np.ndarray] = None

    def _init_windows(self) -> None:
        # Crear ventanas y asociar callbacks de ratón para seleccionar puntos
        cv2.namedWindow(self.window_left)
        cv2.namedWindow(self.window_right)
        cv2.namedWindow(self.window_aux)

        cv2.setMouseCallback(self.window_left, self._on_left_click)
        cv2.setMouseCallback(self.window_right, self._on_right_click)

    def _on_left_click(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        # Al pulsar en la ventana izquierda en modo triangulación, guardar coordenada
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == self.MODE_TRIANGULATION:
            self.left_click = (x, y)
            self.last_3d = None

    def _on_right_click(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        # Al pulsar en la ventana derecha en modo triangulación, guardar y calcular 3D
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == self.MODE_TRIANGULATION:
            self.right_click = (x, y)
            self._try_triangulate()

    def _try_triangulate(self) -> None:
        # Si hay ambos clicks, llamar a StereoTriangulator.triangulate_points
        if self.left_click is None or self.right_click is None:
            return

        pts_3d = self.stereo.triangulate_points([self.left_click], [self.right_click])
        self.last_3d = pts_3d[0]
        print(
            "Triangulación 3D (mm): "
            f"X={self.last_3d[0]:.2f}, Y={self.last_3d[1]:.2f}, Z={self.last_3d[2]:.2f}"
        )

    @staticmethod
    def _draw_header(frame: np.ndarray, text: str) -> np.ndarray:
        # Dibuja una cabecera negra con texto informativo en la parte superior
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    @staticmethod
    def _normalize_disparity(disparity: np.ndarray) -> np.ndarray:
        # Escala la disparidad a 0-255 para visualización
        valid = disparity > 0
        vis = np.zeros_like(disparity, dtype=np.uint8)
        if np.any(valid):
            values = disparity[valid]
            disp_min = float(np.min(values))
            disp_max = float(np.max(values))
            if disp_max > disp_min:
                scaled = ((disparity - disp_min) * 255.0 / (disp_max - disp_min)).clip(0, 255)
                vis = scaled.astype(np.uint8)
        return vis

    def _mode_raw(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        # Modo inicial: mostrar las imágenes tal cual llegan de la cámara
        left_view = self._draw_header(
            frame_left,
            "Modo crudo: pulsa 1 rectificar | 2 triangular | 3 disparidad | 4 gestos | q salir",
        )
        right_view = self._draw_header(frame_right, "Vista derecha en crudo")

        cv2.imshow(self.window_left, left_view)
        cv2.imshow(self.window_right, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_rectification(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        # Modo 1: rectificar y dibujar lineas epipolares para comprobar alineación
        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        epi_left, epi_right = self.stereo.draw_epilines(rect_left, rect_right)

        epi_left = self._draw_header(epi_left, "Modo 1: Comprobacion rectificacion | 1/2/3/4 cambiar modo | q salir")
        epi_right = self._draw_header(epi_right, "Las lineas horizontales deberian coincidir fila a fila")

        cv2.imshow(self.window_left, epi_left)
        cv2.imshow(self.window_right, epi_right)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_manual_triangulation(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        # Modo 2: Triangulación manual por selección de puntos en ambas vistas
        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)

        left_view = rect_left.copy()
        right_view = rect_right.copy()

        if self.left_click is not None:
            cv2.circle(left_view, self.left_click, 7, (0, 255, 0), -1)
        if self.right_click is not None:
            cv2.circle(right_view, self.right_click, 7, (0, 255, 0), -1)

        if self.last_3d is not None:
            txt = f"X={self.last_3d[0]:.1f} Y={self.last_3d[1]:.1f} Z={self.last_3d[2]:.1f} mm"
            cv2.putText(left_view, txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(right_view, txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        left_view = self._draw_header(left_view, "Modo 2: Click IZQ luego DER para triangular")
        right_view = self._draw_header(right_view, "Pulsa 'r' para limpiar puntos seleccionados")

        cv2.imshow(self.window_left, left_view)
        cv2.imshow(self.window_right, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_disparity(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        # Modo 3: cálculo de disparidad densa y visualización
        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        disparity = self.stereo.compute_disparity(rect_left, rect_right)

        disp_vis = self._normalize_disparity(disparity)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        left_view = self._draw_header(rect_left, "Modo 3: Disparidad en tiempo real")
        right_view = self._draw_header(rect_right, "Objetos cercanos aparecen más oscuros en la disparidad")

        cv2.imshow(self.window_left, left_view)
        cv2.imshow(self.window_right, right_view)
        cv2.imshow(self.window_aux, disp_color)

    def _mode_gestures(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        # Modo 4: reconocimiento de gestos con MediaPipe (iniciado bajo demanda)
        if self.gesture is None:
            self.gesture = GestureRecognizer()

        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        left_view, left_gestures = self.gesture.process_frame(rect_left)
        right_view, right_gestures = self.gesture.process_frame(rect_right)

        left_text = "Gestos L: " + (", ".join(left_gestures) if left_gestures else "ninguno")
        right_text = "Gestos R: " + (", ".join(right_gestures) if right_gestures else "ninguno")

        left_view = self._draw_header(left_view, "Modo 4: Gestos (MediaPipe)")
        right_view = self._draw_header(right_view, right_text)
        cv2.putText(left_view, left_text, (10, left_view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(self.window_left, left_view)
        cv2.imshow(self.window_right, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def run(self) -> None:
        # Abrir la cámara y comenzar el bucle principal
        if not self.camera.open():
            raise RuntimeError(
                "No se pudo abrir la cámara ZED2. "
                f"Código SDK: {self.camera.last_open_status}"
            )

        self._init_windows()
        print("Aplicacion estereo iniciada. Teclas: 1 Rectificar | 2 Triangular | 3 Disparidad | 4 Gestos | q Salir")

        try:
            while True:
                # Esperar a que haya un nuevo frame disponible
                if not self.camera.grab():
                    continue

                frame_left, frame_right = self.camera.get_frames()

                # Renderizar según el modo activo
                if self.mode == self.MODE_RAW:
                    self._mode_raw(frame_left, frame_right)
                elif self.mode == self.MODE_RECTIFICATION:
                    self._mode_rectification(frame_left, frame_right)
                elif self.mode == self.MODE_TRIANGULATION:
                    self._mode_manual_triangulation(frame_left, frame_right)
                elif self.mode == self.MODE_DISPARITY:
                    self._mode_disparity(frame_left, frame_right)
                elif self.mode == self.MODE_GESTURES:
                    self._mode_gestures(frame_left, frame_right)

                # Procesar tecla
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("1"):
                    self.mode = self.MODE_RECTIFICATION
                elif key == ord("2"):
                    self.mode = self.MODE_TRIANGULATION
                elif key == ord("3"):
                    self.mode = self.MODE_DISPARITY
                elif key == ord("4"):
                    self.mode = self.MODE_GESTURES
                elif key == ord("r") and self.mode == self.MODE_TRIANGULATION:
                    # Limpiar selección de puntos en modo triangulación
                    self.left_click = None
                    self.right_click = None
                    self.last_3d = None

        finally:
            self.camera.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = StereoInteractiveApp()
    app.run()
