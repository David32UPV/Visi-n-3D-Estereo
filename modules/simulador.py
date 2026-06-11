"""Simulador PyBullet de bin picking lanzado desde la detección estéreo.

Cuando `main.py` confirma que YOLO ha detectado exactamente 5 cajas, manda el
"OK" a este módulo con las coordenadas 3D (X, Y, Z en mm, respecto a la cámara)
de cada caja. Este módulo abre PyBullet en un proceso aparte, carga un cobot
KUKA iiwa y coloca las 5 cajas en posiciones equivalentes a las que ve la ZED.

Además, una vez lanzada la escena, `main.py` sigue enviando las posiciones de
las cajas en cada frame por una cola. El proceso de PyBullet reposiciona cada
caja en tiempo real, de modo que si mueves una caja con la mano delante de la
ZED, su gemela en el simulador se mueve igual.

Decisiones de diseño:
- Proceso separado (`multiprocessing` con `spawn`): el bucle de OpenCV/ZED de
  `main.py` no se bloquea y el hijo no hereda el estado de CUDA/ZED (que no
  tolera bien `fork`).
- Emparejamiento por clase: cada detección actualiza la caja de su misma clase,
  así mover una caja no desplaza a las demás.
- Offset de centrado fijo (calculado al lanzar): mantiene el grupo delante del
  robot sin que una caja arrastre al resto.
- Cajas cinemáticas (masa 0): se reposicionan sin caer ni temblar entre frames.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Configuración de la escena (ajusta aquí sin tocar la lógica)
# ---------------------------------------------------------------------------

# Cobot a cargar. Viene incluido en pybullet_data (7 ejes).
ROBOT_URDF = "kuka_iiwa/model.urdf"
ROBOT_BASE_POSITION = (0.0, 0.0, 0.0)

# Pose de la cámara virtual en el mundo del simulador.
# Por defecto la modelamos como una cámara cenital a `CAMERA_HEIGHT` metros sobre
# el suelo, mirando hacia abajo. La rotación cámara->mundo correspondiente lleva:
#   X_cam (derecha en imagen) -> +X mundo
#   Y_cam (abajo en imagen)   -> -Y mundo
#   Z_cam (profundidad)       -> -Z mundo (hacia el suelo)
CAMERA_HEIGHT = 0.9  # metros
_R_CAM_TO_WORLD = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]],
    dtype=np.float64,
)

# Dónde colocamos el grupo de cajas (su centroide) respecto a la base del robot.
# Así las cajas aparecen delante del cobot y dentro de su alcance.
WORKSPACE_CENTER = (0.6, 0.0)  # (x, y) en metros

# Tamaño por defecto de cada caja (semiejes en metros). No tenemos dimensiones
# por clase fiables, así que usamos una caja genérica y distinguimos por color.
BOX_HALF_EXTENTS = (0.05, 0.04, 0.03)
# Masa 0 => caja cinemática: la conduce la cámara, no la física. Evita que caiga
# o tiemble entre actualizaciones de posición.
BOX_MASS = 0.0

# Color RGBA por clase para identificar visualmente cada tipo de caja.
CLASS_COLORS = {
    "caja_busbar": (0.20, 0.40, 0.95, 1.0),
    "caja_chasis": (0.10, 0.80, 0.80, 1.0),
    "caja_manguera": (0.95, 0.55, 0.20, 1.0),
    "caja_pcb": (0.30, 0.85, 0.45, 1.0),
    "caja_potencia": (0.70, 0.25, 0.75, 1.0),
}
_DEFAULT_COLOR = (0.6, 0.6, 0.6, 1.0)


def cam_to_world(point_mm: Sequence[float]) -> np.ndarray:
    """Transforma un punto (X, Y, Z) en mm del marco de la cámara al mundo (m)."""
    p_m = np.asarray(point_mm, dtype=np.float64) / 1000.0
    world = _R_CAM_TO_WORLD @ p_m
    world[2] += CAMERA_HEIGHT
    return world


def _box_world_xy(point_mm: Sequence[float], offset_xy: np.ndarray) -> tuple[float, float]:
    """Posición (x, y) de una caja en el mundo, aplicando el offset de centrado fijo."""
    world = cam_to_world(point_mm)
    return float(world[0] + offset_xy[0]), float(world[1] + offset_xy[1])


def _populate_scene(p: Any, boxes: Sequence[dict]) -> tuple[dict, np.ndarray]:
    """Carga suelo, robot y cajas. Devuelve (body_por_clase, offset_xy fijo)."""
    import pybullet_data

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF(ROBOT_URDF, ROBOT_BASE_POSITION, useFixedBase=True)

    # Offset de centrado: se calcula UNA vez con la disposición inicial y se
    # reutiliza en todas las actualizaciones, para que mover una caja no
    # desplace al resto del grupo.
    worlds = [cam_to_world(b["world_mm"]) for b in boxes]
    centroid_xy = np.mean([w[:2] for w in worlds], axis=0) if worlds else np.zeros(2)
    offset_xy = np.asarray(WORKSPACE_CENTER, dtype=np.float64) - centroid_xy

    z = float(BOX_HALF_EXTENTS[2])  # apoyada sobre el suelo
    body_by_class: dict[str, int] = {}
    for box in boxes:
        name = box.get("name", "caja")
        x, y = _box_world_xy(box["world_mm"], offset_xy)
        color = CLASS_COLORS.get(name, _DEFAULT_COLOR)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=BOX_HALF_EXTENTS)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=BOX_HALF_EXTENTS, rgbaColor=list(color))
        body_by_class[name] = p.createMultiBody(BOX_MASS, col, vis, basePosition=[x, y, z])

    return body_by_class, offset_xy


def _apply_updates(p: Any, boxes: Sequence[dict], body_by_class: dict, offset_xy: np.ndarray) -> None:
    """Reposiciona en la escena cada caja según su clase (emparejamiento por nombre)."""
    z = float(BOX_HALF_EXTENTS[2])
    for box in boxes:
        body = body_by_class.get(box.get("name", "caja"))
        if body is None:
            continue  # clase no presente en el lanzamiento inicial; la ignoramos
        x, y = _box_world_xy(box["world_mm"], offset_xy)
        p.resetBasePositionAndOrientation(body, [x, y, z], [0.0, 0.0, 0.0, 1.0])


def _run_scene(boxes: Sequence[dict], update_queue: Any = None, gui: bool = True) -> None:
    """Punto de entrada del proceso hijo: abre PyBullet y mantiene la escena viva.

    Mientras la ventana esté abierta, drena `update_queue` y reposiciona las
    cajas con la última actualización disponible. Con `gui=False` (para tests)
    construye la escena en DIRECT, da unos pasos y cierra.
    """
    import pybullet as p

    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    try:
        body_by_class, offset_xy = _populate_scene(p, boxes)

        if not gui:
            for _ in range(10):
                p.stepSimulation()
            return

        import time

        print(f"[SIM] Escena lista con {len(boxes)} cajas y cobot KUKA. Cierra la ventana para terminar.")
        while p.isConnected():
            # Quedarnos solo con la actualización más reciente de la cola.
            if update_queue is not None:
                latest = None
                try:
                    while True:
                        latest = update_queue.get_nowait()
                except queue.Empty:
                    pass
                if latest is not None:
                    _apply_updates(p, latest, body_by_class, offset_xy)

            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        if p.isConnected():
            p.disconnect()


class BinPickingSimulator:
    """Lanza y alimenta en tiempo real la escena PyBullet en un proceso aparte."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._proc: Optional[mp.process.BaseProcess] = None
        self._queue: Optional[Any] = None
        self._launched = False

    @property
    def launched(self) -> bool:
        """True si ya se ha lanzado la escena (solo se lanza una vez)."""
        return self._launched

    @property
    def alive(self) -> bool:
        """True si el proceso del simulador sigue vivo (ventana abierta)."""
        return self._proc is not None and self._proc.is_alive()

    def launch(self, boxes: Sequence[dict]) -> bool:
        """Arranca el simulador con las cajas dadas. Idempotente: solo la 1ª vez.

        Args:
            boxes: lista de dicts con claves `name` (clase) y `world_mm` (X,Y,Z mm).

        Returns:
            True si se lanzó en esta llamada, False si ya estaba lanzado.
        """
        if self._launched:
            return False

        # `spawn` evita heredar el estado de CUDA/ZED del proceso principal.
        self._queue = self._ctx.Queue(maxsize=2)
        self._proc = self._ctx.Process(
            target=_run_scene,
            args=(list(boxes), self._queue),
            daemon=False,
        )
        self._proc.start()
        self._launched = True
        return True

    def update(self, boxes: Sequence[dict]) -> None:
        """Envía las posiciones actuales de las cajas al simulador (no bloqueante).

        Si la cola está llena (el simulador aún procesa el frame anterior) se
        descarta esta actualización: siempre prima la más reciente.
        """
        if not self.alive or self._queue is None:
            return
        try:
            self._queue.put_nowait(list(boxes))
        except queue.Full:
            pass

    def close(self) -> None:
        """Cierra el simulador si sigue vivo (opcional)."""
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)


__all__ = ["BinPickingSimulator", "cam_to_world"]
