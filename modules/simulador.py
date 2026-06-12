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
# Zona de descarga al LATERAL DERECHO del robot (Y negativa = derecha del cobot,
# que mira hacia +X) y algo más apartada, en vez de justo enfrente. Las cajas se
# apilan en COLUMNA: se alinean a lo largo de la profundidad (eje X), una detrás
# de otra alejándose del robot, en lugar de en fila lateral (eje Y).
# Si en tu vista la descarga sale al lado izquierdo, pon la Y de DROP_CENTER en
# positivo. Distancias dentro del alcance del KUKA.
DROP_CENTER = (0.12, -0.4)   # (x, y) en metros: 1ª caja de la columna (lateral derecho)
DROP_SPACING = 0.10          # separación entre cajas ya descargadas (m)
APPROACH_HEIGHT = 0.18       # altura sobre la caja para aproximar/levantar (m)
GRAB_Z_OFFSET = 0.02         # la caja cuelga un poco por debajo del efector (m)
REACH_TOL = 0.05             # distancia para dar un waypoint por alcanzado (m)
MAX_STEPS_PER_WAYPOINT = 400 # tope de seguridad por waypoint (~1.7 s a 240 Hz)

# Pausas explícitas del robot. La idea es que SOLO se pare exactamente 1 s en
# cuatro momentos (prepick, pick, preplace y place) y que el resto de
# transiciones sean sin dwell. El bucle de simulación corre a 240 Hz.
SIM_HZ = 240.0
PAUSE_TICKS = int(round(1.0 * SIM_HZ))  # 1 segundo de pausa
# Detección de "asentado": en vez de esperar el timeout completo en cada
# waypoint (lo que metía ~1.7 s muertos por punto, y el doble cuando dos
# waypoints compartían posición), damos el movimiento por terminado en cuanto el
# efector llega a la tolerancia O deja de acercarse (se queda parado). Así las
# transiciones sin pausa son inmediatas.
STALL_EPS = 0.0015   # m: movimiento del efector por tick por debajo del cual lo
                     # consideramos parado
STALL_TICKS = 30     # ticks parado seguidos para dar el waypoint por asentado

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


def _ordered_body_ids(p: Any, body_by_class: dict) -> list:
    """Ids de TODAS las cajas de la escena, de la más cercana al robot a la más lejana.

    Se ordena por la X de simulación (que codifica la profundidad/Z de la cámara:
    menor X = caja más cercana al robot = caja más cercana a la cámara), usando la
    posición ACTUAL de cada caja en PyBullet.

    Clave del arreglo: tomamos las cajas presentes en la escena
    (`body_by_class`), NO las de la última detección de YOLO. Al enseñar la palma
    abierta la mano suele ocultar o excluir varias cajas, así que el último frame
    de YOLO puede traer solo 1-2 cajas; si nos basáramos en él, el robot solo haría
    el pick&place de esas. Usando la escena completa siempre intenta las 5.
    """
    bodies = [body for body in body_by_class.values() if body is not None]

    def _forward(body: int) -> float:
        pos, _ = p.getBasePositionAndOrientation(body)
        return float(pos[0])  # X simulación = profundidad de la cámara

    return sorted(bodies, key=_forward)


class _BinPickingController:
    """Máquina de estados que mueve el KUKA para hacer pick&place de las cajas.

    Recorre las cajas en el orden dado (por Z ascendente). Para cada una sigue
    una secuencia de waypoints: aproximar por encima, bajar+agarrar, levantar,
    transportar a la zona de descarga, bajar+soltar y retirarse. Mientras la
    caja está "agarrada", su posición sigue al efector final (agarre cinemático).

    El robot solo se detiene exactamente 1 s en cuatro puntos (prepick, pick,
    preplace y place); el resto de movimientos encadenan sin esperas. Cada
    movimiento se da por terminado en cuanto el efector llega o deja de acercarse
    al objetivo, sin agotar el timeout, para que no haya delays sobrantes.
    """

    def __init__(self, robot_id: int, ordered_bodies: Sequence[int]) -> None:
        self.robot = robot_id
        self.pending = list(ordered_bodies)
        self.total = len(self.pending)   # cajas que se deben colocar (objetivo)
        self.placed = 0                  # cajas ya descargadas con éxito
        self.box_z = float(BOX_HALF_EXTENTS[2])
        self.current: Optional[int] = None
        self.waypoints: list = []
        self.wp_index = 0
        self.steps = 0
        self.drop_index = 0
        self.done = False
        self.homing = False              # True mientras vuelve a la pose de inicio
        self._down_orn: Optional[Sequence[float]] = None
        # Estado de la caja agarrada (persistente: la caja sigue al efector
        # mientras esté True; se alterna "al llegar" al waypoint de pick/place).
        self.grabbed = False
        # Sub-estado del waypoint actual: "moving" (yendo al objetivo) u
        # "holding" (parado la pausa de 1 s). Y detección de parada del efector.
        self.phase = "moving"
        self.hold_left = 0
        self._stall = 0
        self._last_ee: Optional[np.ndarray] = None

    def _ee_pos(self, p: Any) -> np.ndarray:
        state = p.getLinkState(self.robot, EE_LINK_INDEX, computeForwardKinematics=True)
        return np.array(state[4], dtype=np.float64)

    def _drop_xy(self, index: int) -> tuple[float, float]:
        # Columna vertical: las cajas se alinean en profundidad (eje X),
        # alejándose del robot; el lateral (Y) se mantiene fijo.
        return (DROP_CENTER[0] + index * DROP_SPACING, DROP_CENTER[1])

    def _plan_box(self, p: Any, body: int, drop_index: int) -> list:
        pos, _ = p.getBasePositionAndOrientation(body)
        bx, by = float(pos[0]), float(pos[1])
        above_box = [bx, by, self.box_z + APPROACH_HEIGHT]
        at_box = [bx, by, self.box_z + GRAB_Z_OFFSET]
        dx, dy = self._drop_xy(drop_index)
        above_drop = [dx, dy, self.box_z + APPROACH_HEIGHT]
        at_drop = [dx, dy, self.box_z + GRAB_Z_OFFSET]
        # (objetivo_efector, grab_al_llegar, pausa_en_ticks)
        #   grab_al_llegar: True -> agarra la caja al asentarse en el waypoint,
        #                   False -> la suelta, None -> no cambia el agarre.
        # No hay dos waypoints consecutivos en la misma posición: el agarre y el
        # soltado ocurren "al llegar" a un único waypoint de bajada, así no se
        # duplica el tiempo de espera. Solo estos cuatro puntos tienen pausa de 1 s.
        return [
            (above_box, None, PAUSE_TICKS),    # prepick: llega y espera 1 s antes de bajar
            (at_box, True, PAUSE_TICKS),       # PICK: baja, agarra y espera 1 s
            (above_box, None, 0),              # levanta (sin pausa)
            (above_drop, None, PAUSE_TICKS),   # preplace: llega y espera 1 s antes de bajar
            (at_drop, False, PAUSE_TICKS),     # PLACE: baja, suelta y espera 1 s
            (above_drop, None, 0),             # se retira (sin pausa)
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

    def _follow_ee(self, p: Any, ee: np.ndarray) -> None:
        """La caja agarrada cuelga del efector final (agarre cinemático)."""
        p.resetBasePositionAndOrientation(
            self.current,
            [float(ee[0]), float(ee[1]), max(self.box_z, float(ee[2]) - GRAB_Z_OFFSET)],
            [0.0, 0.0, 0.0, 1.0],
        )

    def _rest_at_drop(self, p: Any) -> None:
        """Apoya la caja actual en su hueco de la zona de descarga, sobre el suelo."""
        dx, dy = self._drop_xy(self.drop_index - 1)
        p.resetBasePositionAndOrientation(self.current, [dx, dy, self.box_z], [0.0, 0.0, 0.0, 1.0])

    def _send_home(self, p: Any) -> None:
        """Comanda las articulaciones a la pose de reposo inicial del KUKA."""
        for j, angle in enumerate(KUKA_REST_POSES):
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL,
                targetPosition=angle, force=300.0, maxVelocity=3.0,
            )

    def _return_home(self, p: Any) -> None:
        """Tras colocar todas las cajas, vuelve a la pose de inicio y termina."""
        if not self.homing:
            self.homing = True
            self.steps = 0
            self._stall = 0
            self._last_ee = self._ee_pos(p)
            print("[SIM] Bin picking terminado -> volviendo a la pose de inicio.")
        self._send_home(p)
        ee = self._ee_pos(p)
        self.steps += 1
        move = float(np.linalg.norm(ee - self._last_ee)) if self._last_ee is not None else 0.0
        self._stall = self._stall + 1 if move < STALL_EPS else 0
        self._last_ee = ee
        # Termina al llegar a reposo (efector parado) o por seguridad por tiempo.
        if self._stall >= STALL_TICKS or self.steps > MAX_STEPS_PER_WAYPOINT:
            self.done = True
            print("[SIM] Robot de vuelta en la pose de inicio.")

    def step(self, p: Any) -> None:
        """Avanza la rutina un tick de simulación."""
        if self.done:
            return
        if self._down_orn is None:
            self._down_orn = p.getQuaternionFromEuler([0.0, math.pi, 0.0])

        # ¿Empezar con la siguiente caja?
        if self.current is None:
            if not self.pending:
                # No quedan cajas: volver a la pose de inicio antes de terminar.
                self._return_home(p)
                return
            self.current = self.pending.pop(0)
            self.waypoints = self._plan_box(p, self.current, self.drop_index)
            self.drop_index += 1
            self.wp_index = 0
            self.steps = 0
            self.phase = "moving"
            self.hold_left = 0
            self._stall = 0
            self._last_ee = self._ee_pos(p)
            self.grabbed = False  # cada caja arranca sin agarrar

        target, grab_after, pause_ticks = self.waypoints[self.wp_index]
        self._send_ik(p, target)

        ee = self._ee_pos(p)
        # Mientras la caja esté agarrada, sigue al efector (también durante la pausa).
        if self.grabbed:
            self._follow_ee(p, ee)

        if self.phase == "moving":
            self.steps += 1
            # ¿Se ha quedado parado el efector (no se acerca más al objetivo)?
            move = float(np.linalg.norm(ee - self._last_ee)) if self._last_ee is not None else 0.0
            self._stall = self._stall + 1 if move < STALL_EPS else 0
            self._last_ee = ee

            reached = float(np.linalg.norm(ee - np.asarray(target, dtype=np.float64))) < REACH_TOL
            settled = reached or self._stall >= STALL_TICKS or self.steps > MAX_STEPS_PER_WAYPOINT
            if settled:
                # Agarre/soltado "al llegar" al waypoint.
                if grab_after is True:
                    self.grabbed = True
                elif grab_after is False:
                    self.grabbed = False
                    self._rest_at_drop(p)  # la caja se queda apoyada al soltarla
                self.phase = "holding"
                self.hold_left = pause_ticks
        else:  # holding: parado exactamente `pause_ticks` ticks (1 s o 0)
            if self.hold_left > 0:
                self.hold_left -= 1
            else:
                self.wp_index += 1
                self.phase = "moving"
                self.steps = 0
                self._stall = 0
                self._last_ee = ee
                if self.wp_index >= len(self.waypoints):
                    # Caja terminada: dejarla bien apoyada en su sitio de descarga.
                    self._rest_at_drop(p)
                    self.placed += 1
                    self.current = None


def _run_scene(boxes: Sequence[dict], update_queue: Any = None, pause_event: Any = None,
               gui: bool = True) -> None:
    """Punto de entrada del proceso hijo: abre PyBullet y mantiene la escena viva.

    Mientras la ventana esté abierta, drena `update_queue` y reposiciona las
    cajas con la última actualización disponible. Con `gui=False` (para tests)
    construye la escena en DIRECT, da unos pasos y cierra.

    `pause_event` controla la pausa/reanudación con gestos: si está activado
    (puño cerrado) la simulación se congela en el sitio (no se avanza la física
    y el robot se queda exactamente donde esté); al desactivarse (palma abierta)
    se reanuda desde ese mismo punto, conservando el estado del bin picking.
    """
    import pybullet as p

    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    try:
        body_by_class, ref_xy, robot_id = _populate_scene(p, boxes)
        last_boxes = list(boxes)
        controller: Optional[_BinPickingController] = None
        pick_done_logged = False
        paused = False

        if not gui:
            for _ in range(10):
                p.stepSimulation()
            return

        import time

        print(f"[SIM] Escena lista con {len(boxes)} cajas y cobot KUKA. Cierra la ventana para terminar.")
        while p.isConnected():
            # Pausa/reanudación con gestos: puño cerrado congela la simulación
            # donde esté el robot; palma abierta la reanuda desde ese mismo punto.
            if pause_event is not None and pause_event.is_set():
                if not paused:
                    paused = True
                    # Congelar el robot exactamente donde esté (cualquier pose).
                    for j in range(p.getNumJoints(robot_id)):
                        pos_j = p.getJointState(robot_id, j)[0]
                        p.resetJointState(robot_id, j, pos_j)  # velocidad a cero
                        p.setJointMotorControl2(
                            robot_id, j, p.POSITION_CONTROL,
                            targetPosition=pos_j, force=300.0,  # mantiene la pose
                        )
                    print("[SIM] STOP (puño cerrado) -> simulación EN PAUSA. Abre la palma para reanudar.")
                time.sleep(1.0 / 240.0)
                continue
            elif paused:
                # El evento se ha desactivado (palma abierta) -> reanudar.
                paused = False
                print("[SIM] Palma abierta -> simulación REANUDADA desde donde se paró.")

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

            # Palma abierta -> arrancar el bin picking (una sola vez). Se cogen
            # TODAS las cajas de la escena, no solo las de la última detección.
            if pick_requested and controller is None:
                ordered = _ordered_body_ids(p, body_by_class)
                if ordered:
                    controller = _BinPickingController(robot_id, ordered)
                    print(f"[SIM] Palma abierta -> bin picking de {len(ordered)} cajas (cercana -> lejana)")
                else:
                    print("[SIM][ERROR] Palma abierta pero no hay cajas en la escena: no se puede hacer pick and place.")

            if controller is not None and not controller.done:
                try:
                    controller.step(p)
                except Exception as exc:  # noqa: BLE001 - no queremos cerrar la escena
                    print("[SIM][ERROR] Excepción en bin picking, se detiene la rutina:", exc)
                    controller.done = True

                # Al terminar la rutina, confirmar que se han colocado las 5 cajas.
                if controller.done and not pick_done_logged:
                    pick_done_logged = True
                    if controller.placed >= controller.total:
                        print(f"[SIM] Bin picking completado: {controller.placed}/{controller.total} cajas colocadas.")
                    else:
                        print(
                            f"[SIM][ERROR] Bin picking incompleto: solo {controller.placed}/{controller.total} "
                            "cajas colocadas. El robot no terminó el pick and place de las 5 cajas "
                            "(revisa alcance/IK del KUKA o las posiciones de las cajas)."
                        )

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
        self._pause_event: Optional[Any] = None
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
        # Event aparte para la pausa: es una señal de seguridad y no puede perderse
        # si la cola de updates está llena (la cola usa put_nowait y descarta).
        # Activado = pausa (puño); desactivado = en marcha (palma).
        self._pause_event = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_run_scene,
            args=(list(boxes), self._queue, self._pause_event),
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

    def pause(self) -> None:
        """Pausa la simulación (puño cerrado): el robot se congela donde esté.

        Es seguro llamarlo en cada frame mientras el puño siga cerrado. Usa un
        Event, así que la señal nunca se pierde (a diferencia de la cola).
        """
        if self._pause_event is not None:
            self._pause_event.set()

    def resume(self) -> None:
        """Reanuda la simulación (palma abierta) desde donde se quedó pausada.

        Es seguro llamarlo en cada frame: si no estaba en pausa no hace nada.
        """
        if self._pause_event is not None:
            self._pause_event.clear()

    def close(self) -> None:
        """Cierra el simulador si sigue vivo (opcional)."""
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)


__all__ = ["BinPickingSimulator", "cam_to_world"]
