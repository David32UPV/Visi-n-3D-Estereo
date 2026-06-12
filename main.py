"""Aplicación interactiva para validación de rectificación, triangulación y gestos.

Modos disponibles:
1 - Comprobación de rectificación (dibujar líneas epipolares horizontales)
2 - Triangulación manual por click (click mitad izquierda -> click mitad derecha)
3 - Mapa de disparidad en tiempo real (SGBM)
4 - Reconocimiento de gestos (MediaPipe)
5 - Detección automática de cajas con YOLOv8-seg
6 - YOLO + Gestos simultáneos (modo por defecto)

Controles:
- Teclas 1/2/3/4/5/6: cambiar modo
- q: salir
- r: limpiar puntos seleccionados (modo 2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from modules.camera_module import ZEDCamera
from modules.gesture_module import GestureRecognizer
from modules.simulador import BinPickingSimulator
from modules.stereo_module import StereoTriangulator
from modules.yolo_stereo_module import YoloSegBoxModule


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
    MODE_YOLO_BOXES = 5
    MODE_YOLO_GESTURES = 6

    # Factor de escala para la ventana principal (0.5 = mitad de tamaño).
    # Cámbialo para ajustar el tamaño de visualización.
    DISPLAY_SCALE = 0.8

    # Nº de cajas que debe contar YOLO para lanzar el simulador PyBullet.
    EXPECTED_BOXES = 5

    @staticmethod
    def _find_recursive_path(root: Path, filename: str, preferred_parent: str | None = None) -> Optional[Path]:
        matches = [path for path in root.rglob(filename) if path.is_file()]
        if preferred_parent is not None:
            for path in matches:
                if preferred_parent in {parent.name for parent in path.parents}:
                    return path
        return matches[0] if matches else None

    @classmethod
    def _resolve_dataset_yaml(cls, base_dir: Path) -> Path:
        preferred = base_dir / "etiquetado_5_tipos_de_cajas.yolov8" / "data.yaml"
        if preferred.exists():
            return preferred

        fallback = cls._find_recursive_path(base_dir, "data.yaml", preferred_parent="etiquetado_5_tipos_de_cajas.yolov8")
        if fallback is not None:
            return fallback

        return preferred

    @classmethod
    def _resolve_yolo_weights(cls, base_dir: Path) -> Path:
        preferred = base_dir / "runs" / "yolo_boxes" / "boxes_seg" / "weights" / "best.pt"
        if preferred.exists():
            return preferred

        fallback = cls._find_recursive_path(base_dir, "best.pt", preferred_parent="boxes_seg")
        if fallback is not None:
            return fallback

        return preferred

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        calib_path = base_dir / "calibration" / "stereo_calib.npz"
        dataset_yaml = self._resolve_dataset_yaml(base_dir)

        if not calib_path.exists():
            raise FileNotFoundError(f"Fichero de calibración no encontrado: {calib_path}")
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"No existe el dataset YAML: {dataset_yaml}")

        # Inicializaciones principales
        self.camera = ZEDCamera()
        self.stereo = StereoTriangulator(calib_path=str(calib_path), image_size=(1280, 720))
        self.gesture: Optional[GestureRecognizer] = None
        self.yolo: Optional[YoloSegBoxModule] = None
        self.simulator = BinPickingSimulator()
        self.yolo_dataset_yaml = dataset_yaml
        self.yolo_weights = self._resolve_yolo_weights(base_dir)
        self.yolo_project_dir = self.yolo_weights.parent.parent.parent if self.yolo_weights.exists() else base_dir / "runs" / "yolo_boxes"

        # Estado del modo y ventanas
        self.mode = self.MODE_YOLO_GESTURES
        self.window_main = "ZED2 Stereo"
        self.window_aux = "ZED2 Disparity"

        # Variables para triangulación manual
        self.left_click: Optional[Tuple[int, int]] = None
        self.right_click: Optional[Tuple[int, int]] = None
        self.last_3d: Optional[np.ndarray] = None

    def _ensure_gesture(self) -> GestureRecognizer:
        if self.gesture is None:
            self.gesture = GestureRecognizer()
        return self.gesture

    def _ensure_yolo(self) -> YoloSegBoxModule:
        if self.yolo is None:
            self.yolo = YoloSegBoxModule(
                dataset_yaml=self.yolo_dataset_yaml,
                stereo=self.stereo,
                project_dir=self.yolo_project_dir,
                custom_weights=self.yolo_weights,
            )
            self.yolo.load_trained_model()
        return self.yolo

    def _init_windows(self) -> None:
        cv2.namedWindow(self.window_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_aux, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_main, self._on_stereo_click)

    def _on_stereo_click(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.mode != self.MODE_TRIANGULATION:
            return
        # Revertir la escala de visualización para obtener coordenadas en el frame original
        s = self.DISPLAY_SCALE
        x_orig = int(x / s)
        y_orig = int(y / s)
        frame_w = self.stereo.image_size[0]
        if x_orig < frame_w:
            self.left_click = (x_orig, y_orig)
            self.last_3d = None
        else:
            self.right_click = (x_orig - frame_w, y_orig)
            self._try_triangulate()

    def _try_triangulate(self) -> None:
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
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    @staticmethod
    def _normalize_disparity(disparity: np.ndarray) -> np.ndarray:
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

    def _show_stereo(self, left_view: np.ndarray, right_view: np.ndarray) -> None:
        combined = np.hstack([left_view, right_view])
        if self.DISPLAY_SCALE != 1.0:
            combined = cv2.resize(combined, (0, 0), fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE)
        cv2.imshow(self.window_main, combined)

    def _mode_raw(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        keys_hint = "1 rectificar | 2 triangular | 3 disparidad | 4 gestos | 5 YOLO | 6 YOLO+Gestos | q salir"
        left_view = self._draw_header(frame_left, f"Izquierda — {keys_hint}")
        right_view = self._draw_header(frame_right, "Derecha")
        self._show_stereo(left_view, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_rectification(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        epi_left, epi_right = self.stereo.draw_epilines(rect_left, rect_right)
        epi_left = self._draw_header(epi_left, "Modo 1: Rectificacion | q salir")
        epi_right = self._draw_header(epi_right, "Las lineas horizontales deben coincidir fila a fila")
        self._show_stereo(epi_left, epi_right)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_manual_triangulation(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
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

        left_view = self._draw_header(left_view, "Modo 2: Click mitad IZQ luego DER para triangular")
        right_view = self._draw_header(right_view, "Pulsa 'r' para limpiar puntos")
        self._show_stereo(left_view, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_disparity(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        disparity = self.stereo.compute_disparity(rect_left, rect_right)

        disp_vis = self._normalize_disparity(disparity)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        left_view = self._draw_header(rect_left, "Modo 3: Disparidad en tiempo real")
        right_view = self._draw_header(rect_right, "Objetos cercanos aparecen mas oscuros")
        self._show_stereo(left_view, right_view)
        cv2.imshow(self.window_aux, disp_color)

    def _mode_gestures(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        self._ensure_gesture()

        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        left_view, left_gestures = self.gesture.process_frame(rect_left)
        right_view, right_gestures = self.gesture.process_frame(rect_right)

        left_text = "Gestos L: " + (", ".join(left_gestures) if left_gestures else "ninguno")
        right_text = "Gestos R: " + (", ".join(right_gestures) if right_gestures else "ninguno")

        left_view = self._draw_header(left_view, f"Modo 4: Gestos (MediaPipe) | {left_text}")
        right_view = self._draw_header(right_view, right_text)
        self._show_stereo(left_view, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _mode_yolo_boxes(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        detections: list = []
        try:
            yolo = self._ensure_yolo()
            _, left_view, _, right_view, detections = yolo.predict_pair_with_depth(frame_left, frame_right)
            left_view = self._draw_header(left_view, f"Modo 5: YOLO + centroide + Q | detecciones: {len(detections)}")
            right_view = self._draw_header(right_view, "Modo 5: YOLO lente derecha")
        except (FileNotFoundError, RuntimeError) as exc:
            left_view = self._draw_header(frame_left, str(exc))
            right_view = self._draw_header(frame_right, "Comprueba best.pt y calibracion")

        # Lanza el simulador con 5 cajas y luego actualiza sus posiciones en vivo.
        self._sync_simulator(detections)

        self._show_stereo(left_view, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def _prepare_sim_boxes(self, detections: list) -> list:
        """Construye la lista de cajas (clase + X/Y/Z en mm) para el simulador.

        Usa la coordenada 3D ya calculada (`world`). Si alguna caja no tiene
        profundidad válida, estima su Z con la mediana de las válidas y recupera
        X/Y a partir del píxel del centroide y los intrínsecos de la matriz Q.
        """
        Q = self.stereo.Q
        cx, cy, f = -float(Q[0, 3]), -float(Q[1, 3]), float(Q[2, 3])

        valid_z = [d["world"][2] for d in detections if d.get("world") is not None]
        fallback_z = float(np.median(valid_z)) if valid_z else 600.0

        boxes = []
        for detection in detections:
            world = detection.get("world")
            if world is None:
                u, v = detection["center"]
                z = fallback_z
                world = (z * (u - cx) / f, z * (v - cy) / f, z)
            boxes.append(
                {
                    "name": detection.get("class_name", "caja"),
                    "world_mm": (float(world[0]), float(world[1]), float(world[2])),
                }
            )
        return boxes

    def _sync_simulator(self, detections: list) -> None:
        """Lanza el simulador al contar 5 cajas y luego sigue actualizando posiciones.

        - Mientras no esté lanzado: arranca PyBullet cuando hay EXPECTED_BOXES.
        - Una vez lanzado: envía cada frame las posiciones para que las cajas del
          simulador sigan en tiempo real a las cajas reales (movidas con la mano).
        """
        if not detections:
            return

        boxes = self._prepare_sim_boxes(detections)

        if not self.simulator.launched:
            if len(detections) == self.EXPECTED_BOXES and self.simulator.launch(boxes):
                print(f"[SIM] {self.EXPECTED_BOXES} cajas detectadas -> lanzando simulador PyBullet con cobot KUKA")
        else:
            self.simulator.update(boxes)

    def _mode_yolo_and_gestures(self, frame_left: np.ndarray, frame_right: np.ndarray) -> None:
        gesture = self._ensure_gesture()

        # 1) Detectar manos sobre los frames RECTIFICADOS (mismo sistema que YOLO),
        #    sin dibujar todavia, para obtener sus hitboxes como zonas de exclusion.
        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        hands_left = gesture.detect(rect_left)
        hands_right = gesture.detect(rect_right)
        exclude_left = [hand["bbox"] for hand in hands_left]
        exclude_right = [hand["bbox"] for hand in hands_right]

        # 2) YOLO ignora las detecciones que caen sobre las manos.
        try:
            yolo = self._ensure_yolo()
            _, left_view, _, right_view, detections = yolo.predict_pair_with_depth(
                frame_left, frame_right, exclude_left=exclude_left, exclude_right=exclude_right
            )
        except (FileNotFoundError, RuntimeError) as exc:
            left_view = self._draw_header(rect_left, str(exc))
            right_view = self._draw_header(rect_right, "Comprueba best.pt y calibracion")
            detections = []

        # Lanza el simulador con 5 cajas y luego actualiza sus posiciones en vivo.
        self._sync_simulator(detections)

        # 3) Dibujar los gestos encima de la imagen ya anotada por YOLO.
        left_view, left_gestures = gesture.draw(left_view, hands_left)
        right_view, right_gestures = gesture.draw(right_view, hands_right)

        # Palma abierta -> empezar el bin picking y, si estaba en pausa, reanudarlo.
        if self.simulator.launched and ("Open_Palm" in left_gestures or "Open_Palm" in right_gestures):
            self.simulator.start_bin_picking()
            self.simulator.resume()

        # Puño cerrado -> pausar la simulación (el robot se congela donde esté).
        if self.simulator.launched and ("Closed_Fist" in left_gestures or "Closed_Fist" in right_gestures):
            self.simulator.pause()

        left_text = "Gestos: " + (", ".join(left_gestures) if left_gestures else "ninguno")
        left_view = self._draw_header(
            left_view,
            f"Modo 6: YOLO+Gestos | cajas: {len(detections)} | {left_text}",
        )
        right_view = self._draw_header(right_view, "Modo 6: YOLO+Gestos lente derecha")
        self._show_stereo(left_view, right_view)
        cv2.imshow(self.window_aux, np.zeros((400, 600), dtype=np.uint8))

    def run(self) -> None:
        if not self.camera.open():
            raise RuntimeError(
                "No se pudo abrir la cámara ZED2. "
                f"Código SDK: {self.camera.last_open_status}"
            )

        self._init_windows()
        self._ensure_gesture()
        self._ensure_yolo()
        print(
            "Aplicacion estereo iniciada. Arranque automatico en modo YOLO+Gestos. "
            "Teclas: 1 Rectificar | 2 Triangular | 3 Disparidad | 4 Gestos | 5 YOLO | 6 YOLO+Gestos | q Salir"
        )

        try:
            while True:
                if not self.camera.grab():
                    continue

                frame_left, frame_right = self.camera.get_frames()

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
                elif self.mode == self.MODE_YOLO_BOXES:
                    self._mode_yolo_boxes(frame_left, frame_right)
                elif self.mode == self.MODE_YOLO_GESTURES:
                    self._mode_yolo_and_gestures(frame_left, frame_right)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("1"):
                    self.mode = self.MODE_RECTIFICATION
                elif key == ord("2"):
                    self.mode = self.MODE_TRIANGULATION
                elif key == ord("3"):
                    self.mode = self.MODE_DISPARITY
                elif key == ord("4"):
                    self.mode = self.MODE_GESTURES
                elif key == ord("5"):
                    self.mode = self.MODE_YOLO_BOXES
                elif key == ord("6"):
                    self.mode = self.MODE_YOLO_GESTURES
                elif key == ord("r") and self.mode == self.MODE_TRIANGULATION:
                    self.left_click = None
                    self.right_click = None
                    self.last_3d = None

        finally:
            self.camera.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = StereoInteractiveApp()
    app.run()
