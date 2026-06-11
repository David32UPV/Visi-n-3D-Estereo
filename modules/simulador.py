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

import math
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
WORKSPACE_CENTER = (0.45, 0.0)  # (x, y) en metros
# Escala de la separación entre cajas respecto a la real (1.0 = distancias
# reales). Un poco >1 para que no se toquen, conservando la disposición.
SPREAD_SCALE = 1.3

# Tamaño por defecto de cada caja (semiejes en metros). No tenemos dimensiones
# por clase fiables, así que usamos una caja genérica y distinguimos por color.
# Más pequeñas que la separación de descarga (DROP_SPACING) para que no se solapen.
BOX_HALF_EXTENTS = (0.055, 0.045, 0.04)
# Masa 0 => caja cinemática: la conduce la cámara, no la física. Evita que caiga
# o tiemble entre actualizaciones de posición.
BOX_MASS = 0.0

# --- Bin picking (se activa al detectar la palma abierta) ---
EE_LINK_INDEX = 6            # último link del KUKA iiwa (efector final)
DROP_CENTER = (0.25, -0.3)   # zona de descarga, más cerca del robot que las cajas
DROP_SPACING = 0.15          # separación entre cajas ya descargadas (m)
APPROACH_HEIGHT = 0.18       # altura sobre la caja para aproximar/levantar (m)
GRAB_Z_OFFSET = 0.02         # la caja cuelga un poco por debajo del efector (m)
REACH_TOL = 0.05             # distancia para dar un waypoint por alcanzado (m)
MAX_STEPS_PER_WAYPOINT = 400 # tope de seguridad por waypoint (~1.7 s a 240 Hz)

# Límites articulares, rangos y pose de reposo del KUKA iiwa (7 ejes). Se pasan
# a la cinemática inversa para obtener soluciones estables y alcanzables (modo
# null-space) en vez de configuraciones extrañas que no llegan al objetivo.
KUKA_LOWER = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
KUKA_UPPER = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]
KUKA_RANGES = [hi - lo for hi, lo in zip(KUKA_UPPER, KUKA_LOWER)]
KUKA_REST_POSES = [0.0, 0.5, 0.0, -1.4, 0.0, 1.2, 0.0]

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


def _box_floor_xy(point_mm: Sequence[float], ref_xy: Sequence[float]) -> tuple[float, float]:
    """Posición (x, y) de una caja sobre el suelo del simulador.

    El suelo se forma con los dos ejes de la cámara en los que se reparten las
    cajas que ve la ZED:
    - eje lateral  = X de la cámara (izquierda-derecha) -> lado del robot (y),
    - eje profundo = Z de la cámara (cerca-lejos)       -> delante del robot (x).

    (La Y de la cámara, casi vertical en una escena sobre el suelo, no se usa
    para la posición en planta.) Se centra el grupo en `WORKSPACE_CENTER` y se
    escala con `SPREAD_SCALE`, conservando la disposición real.
    """
    lateral = float(point_mm[0]) / 1000.0   # X cámara
    forward = float(point_mm[2]) / 1000.0   # Z cámara (profundidad)
    sim_x = WORKSPACE_CENTER[0] + (forward - ref_xy[1]) * SPREAD_SCALE
    sim_y = WORKSPACE_CENTER[1] + (lateral - ref_xy[0]) * SPREAD_SCALE
    return float(sim_x), float(sim_y)


def _placement_reference(boxes: Sequence[dict]) -> tuple[float, float]:
    """Centroide (lateral, profundidad) en metros del grupo de cajas."""
    if not boxes:
        return (0.0, 0.0)
    laterals = [float(b["world_mm"][0]) / 1000.0 for b in boxes]
    forwards = [float(b["world_mm"][2]) / 1000.0 for b in boxes]
    return (float(np.mean(laterals)), float(np.mean(forwards)))


def _populate_scene(p: Any, boxes: Sequence[dict]) -> tuple[dict, tuple, int]:
    """Carga suelo, robot y cajas. Devuelve (body_por_clase, ref_xy, robot_id)."""
    import pybullet_data

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(ROBOT_URDF, ROBOT_BASE_POSITION, useFixedBase=True)

    # Pose de reposo del robot: arranca en una configuración cómoda para la IK.
    for joint, angle in enumerate(KUKA_REST_POSES):
        p.resetJointState(robot_id, joint, angle)

    # Referencia fija (centroide lateral/profundidad) para centrar el grupo.
    ref_xy = _placement_reference(boxes)

    z = float(BOX_HALF_EXTENTS[2])  # apoyada sobre el suelo
    label_z = z + BOX_HALF_EXTENTS[2] + 0.04  # etiqueta justo encima de la caja
    body_by_class: dict[str, int] = {}
    for box in boxes:
        name = box.get("name", "caja")
        x, y = _box_floor_xy(box["world_mm"], ref_xy)
        color = CLASS_COLORS.get(name, _DEFAULT_COLOR)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=BOX_HALF_EXTENTS)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=BOX_HALF_EXTENTS, rgbaColor=list(color))
        body = p.createMultiBody(BOX_MASS, col, vis, basePosition=[x, y, z])
        # Las cajas son cinemáticas: las movemos nosotros, así que desactivamos
        # sus colisiones para que el brazo no se atasque ni las tire al pasar.
        p.setCollisionFilterGroupMask(body, -1, 0, 0)
        # Etiqueta con el nombre de la clase YOLO, anclada a la caja (la sigue).
        p.addUserDebugText(
            name, [0.0, 0.0, label_z],
            textColorRGB=[0.0, 0.0, 0.0], textSize=1.4,
            parentObjectUniqueId=body,
        )
        body_by_class[name] = body

    return body_by_class, ref_xy, robot_id


def _apply_updates(p: Any, boxes: Sequence[dict], body_by_class: dict, ref_xy: Sequence[float]) -> None:
    """Reposiciona en la escena cada caja según su clase (emparejamiento por nombre)."""
    z = float(BOX_HALF_EXTENTS[2])
    for box in boxes:
        body = body_by_class.get(box.get("name", "caja"))
        if body is None:
            continue  # clase no presente en el lanzamiento inicial; la ignoramos
        x, y = _box_floor_xy(box["world_mm"], ref_xy)
        p.resetBasePositionAndOrientation(body, [x, y, z], [0.0, 0.0, 0.0, 1.0])


def _ordered_body_ids(body_by_class: dict, boxes: Sequence[dict]) -> list:
    """Ids de las cajas ordenadas por Z (profundidad al centroide) ascendente.

    El robot coge primero la caja con menor Z (más cercana) y termina con la de
    mayor Z (más lejana), tal y como se pide.
    """
    ordered = sorted(boxes, key=lambda b: float(b["world_mm"][2]))
    body_ids = [body_by_class.get(b.get("name", "caja")) for b in ordered]
    return [body for body in body_ids if body is not None]


class _BinPickingController:
    """Máquina de estados que mueve el KUKA para hacer pick&place de las cajas.

    Recorre las cajas en el orden dado (por Z ascendente). Para cada una sigue
    una secuencia de waypoints: aproximar por encima, bajar, agarrar, levantar,
    transportar a la zona de descarga, bajar, soltar y retirarse. Mientras la
    caja está "agarrada", su posición sigue al efector final (agarre cinemático).
    """

    def __init__(self, robot_id: int, ordered_bodies: Sequence[int]) -> None:
        self.robot = robot_id
        self.pending = list(ordered_bodies)
        self.box_z = float(BOX_HALF_EXTENTS[2])
        self.current: Optional[int] = None
        self.waypoints: list = []
        self.wp_index = 0
        self.steps = 0
        self.drop_index = 0
        self.done = False
        self._down_orn: Optional[Sequence[float]] = None

    def _ee_pos(self, p: Any) -> np.ndarray:
        state = p.getLinkState(self.robot, EE_LINK_INDEX, computeForwardKinematics=True)
        return np.array(state[4], dtype=np.float64)

    def _drop_xy(self, index: int) -> tuple[float, float]:
        return (DROP_CENTER[0], DROP_CENTER[1] + index * DROP_SPACING)

    def _plan_box(self, p: Any, body: int, drop_index: int) -> list:
        pos, _ = p.getBasePositionAndOrientation(body)
        bx, by = float(pos[0]), float(pos[1])
        above_box = [bx, by, self.box_z + APPROACH_HEIGHT]
        at_box = [bx, by, self.box_z + GRAB_Z_OFFSET]
        dx, dy = self._drop_xy(drop_index)
        above_drop = [dx, dy, self.box_z + APPROACH_HEIGHT]
        at_drop = [dx, dy, self.box_z + GRAB_Z_OFFSET]
        # (objetivo_efector, caja_agarrada)
        return [
            (above_box, False),   # aproximar por encima
            (at_box, False),      # bajar a la caja
            (at_box, True),       # agarrar
            (above_box, True),    # levantar
            (above_drop, True),   # transportar sobre la descarga
            (at_drop, True),      # bajar a soltar
            (at_drop, False),     # soltar
            (above_drop, False),  # retirarse
        ]

    def _send_ik(self, p: Any, target: Sequence[float]) -> None:
        joints = p.calculateInverseKinematics(
            self.robot, EE_LINK_INDEX, list(target), self._down_orn,
            lowerLimits=KUKA_LOWER, upperLimits=KUKA_UPPER,
            jointRanges=KUKA_RANGES, restPoses=KUKA_REST_POSES,
            maxNumIterations=100, residualThreshold=1e-4,
        )
        for j in range(min(len(joints), 7)):
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL,
                targetPosition=joints[j], force=300.0, maxVelocity=3.0,
            )

    def step(self, p: Any) -> None:
        """Avanza la rutina un tick de simulación."""
        if self.done:
            return
        if self._down_orn is None:
            self._down_orn = p.getQuaternionFromEuler([0.0, math.pi, 0.0])

        # ¿Empezar con la siguiente caja?
        if self.current is None:
            if not self.pending:
                self.done = True
                return
            self.current = self.pending.pop(0)
            self.waypoints = self._plan_box(p, self.current, self.drop_index)
            self.drop_index += 1
            self.wp_index = 0
            self.steps = 0

        target, grabbed = self.waypoints[self.wp_index]
        self._send_ik(p, target)

        # Si la caja está agarrada, que siga al efector final.
        if grabbed:
            ee = self._ee_pos(p)
            p.resetBasePositionAndOrientation(
                self.current,
                [float(ee[0]), float(ee[1]), max(self.box_z, float(ee[2]) - GRAB_Z_OFFSET)],
                [0.0, 0.0, 0.0, 1.0],
            )

        self.steps += 1
        reached = np.linalg.norm(self._ee_pos(p) - np.asarray(target, dtype=np.float64)) < REACH_TOL
        if reached or self.steps > MAX_STEPS_PER_WAYPOINT:
            self.wp_index += 1
            self.steps = 0
            if self.wp_index >= len(self.waypoints):
                # Caja terminada: dejarla bien apoyada en su sitio de descarga.
                dx, dy = self._drop_xy(self.drop_index - 1)
                p.resetBasePositionAndOrientation(self.current, [dx, dy, self.box_z], [0.0, 0.0, 0.0, 1.0])
                self.current = None


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
        body_by_class, ref_xy, robot_id = _populate_scene(p, boxes)
        last_boxes = list(boxes)
        controller: Optional[_BinPickingController] = None

        if not gui:
            for _ in range(10):
                p.stepSimulation()
            return

        import time

        print(f"[SIM] Escena lista con {len(boxes)} cajas y cobot KUKA. Cierra la ventana para terminar.")
        while p.isConnected():
            # Vaciar la cola: quedarnos con la última actualización y ver si se
            # ha pedido iniciar el bin picking.
            latest_update = None
            pick_requested = False
            if update_queue is not None:
                try:
                    while True:
                        msg = update_queue.get_nowait()
                        if msg.get("cmd") == "pick":
                            pick_requested = True
                        elif msg.get("cmd") == "update":
                            latest_update = msg.get("boxes")
                except queue.Empty:
                    pass

            if latest_update:
                last_boxes = latest_update
                # Mientras el robot no esté trabajando, las cajas siguen a la ZED.
                if controller is None:
                    _apply_updates(p, latest_update, body_by_class, ref_xy)

            # Palma abierta -> arrancar el bin picking (una sola vez).
            if pick_requested and controller is None and last_boxes:
                ordered = _ordered_body_ids(body_by_class, last_boxes)
                if ordered:
                    controller = _BinPickingController(robot_id, ordered)
                    print("[SIM] Palma abierta -> bin picking por Z ascendente (cercana -> lejana)")

            if controller is not None and not controller.done:
                try:
                    controller.step(p)
                except Exception as exc:  # noqa: BLE001 - no queremos cerrar la escena
                    print("[SIM] Error en bin picking, se detiene la rutina:", exc)
                    controller.done = True

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
            self._queue.put_nowait({"cmd": "update", "boxes": list(boxes)})
        except queue.Full:
            pass

    def start_bin_picking(self) -> None:
        """Manda el OK para que el robot empiece el bin picking (palma abierta).

        Es seguro llamarlo en cada frame mientras la palma siga abierta: el
        simulador solo inicia la rutina la primera vez y luego ignora el resto.
        """
        if not self.alive or self._queue is None:
            return
        try:
            self._queue.put_nowait({"cmd": "pick"})
        except queue.Full:
            pass

    def close(self) -> None:
        """Cierra el simulador si sigue vivo (opcional)."""
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)


__all__ = ["BinPickingSimulator", "cam_to_world"]
