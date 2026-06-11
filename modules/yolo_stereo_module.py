"""Utilidades YOLOv8-seg para entrenar y detectar cajas en dos frames.

Este modulo esta pensado como una pieza aislada de deteccion. No contiene
ninguna logica de rectificacion, disparidad ni triangulacion. Su trabajo se
divide en cuatro partes muy concretas:

1. Leer y validar el `data.yaml` exportado por Roboflow.
2. Crear un split local de entrenamiento/validacion si el export no trae `val`.
3. Entrenar un modelo Ultralytics YOLOv8-seg con tus clases `caja_1` y `caja_2`.
4. Ejecutar inferencia sobre una imagen o sobre dos frames independientes.

El nombre del archivo se mantiene para encajar con el plan del proyecto, pero
la implementacion no depende de disparidad ni triangulacion.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class YoloSegBoxModule:
    """Encapsula el ciclo de vida del detector YOLO de cajas.

    Esta clase sirve para tres momentos distintos del flujo:

    - preparacion del dataset: leer `data.yaml`, resolver rutas y generar una
      validacion local si hace falta;
    - entrenamiento: lanzar YOLOv8-seg sobre el dataset Roboflow;
    - inferencia: cargar el mejor checkpoint y ejecutar deteccion sobre una o
      dos imagenes.

    La clase no conoce nada de ZED, ni de disparidad, ni de triangulacion. Solo
    sabe trabajar con frames de entrada y devolver resultados de YOLO.

    Args:
        dataset_yaml: Ruta al `data.yaml` exportado por Roboflow.
        weights: Pesos base a usar al entrenar o a cargar si no se especifica
            un modelo ya entrenado.
        project_dir: Carpeta donde Ultralytics guardara los runs.
        run_name: Nombre del entrenamiento o inferencia.
        split_fraction: Fraccion de imagenes que se reservara para validacion
            si el dataset no trae una particion valida.
        seed: Semilla para el split reproducible.
    """

    def __init__(
        self,
        dataset_yaml: str | Path,
        weights: str = "yolov8n-seg.pt",
        custom_weights: str | Path | None = None,
        stereo: Any | None = None,
        project_dir: str | Path = "runs/yolo_boxes",
        run_name: str = "boxes_seg",
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        # Guardamos las rutas y parametros para no repetirlos en cada funcion.
        self.dataset_yaml = Path(dataset_yaml)
        self.dataset_root = self.dataset_yaml.parent
        self.weights = weights
        self.custom_weights = Path(custom_weights) if custom_weights is not None else None
        self.stereo = stereo
        self.project_dir = Path(project_dir)
        self.run_name = run_name
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        split_total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError("`train_fraction + val_fraction + test_fraction` debe sumar 1.0.")

        # Cargamos la configuracion para validar clases y rutas desde el principio.
        self.class_names = self.load_dataset_config()["names"]

        # Se rellena cuando se carga o entrena un modelo.
        self.model: Any = None

    @staticmethod
    def _bbox_center(bbox: np.ndarray | list[float] | tuple[float, ...]) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _mask_centroid_from_polygon(mask_polygon: Any) -> Optional[tuple[float, float]]:
        """Calcula el centroide de una máscara poligonal usando cv2.moments."""
        if mask_polygon is None:
            return None

        polygon = np.asarray(mask_polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            return None

        contour = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-6:
            return None

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        return cx, cy

    @staticmethod
    def _result_name_map(result: Any) -> dict[int, str]:
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        return {}

    def _extract_detections(self, result: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        masks = getattr(result, "masks", None)
        masks_xy = getattr(masks, "xy", None) if masks is not None else None

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy), dtype=np.float32)
        class_ids = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), dtype=np.float32)
        names = self._result_name_map(result)

        detections: list[dict[str, Any]] = []
        for index, bbox in enumerate(xyxy):
            center_from_mask = None
            if masks_xy is not None and index < len(masks_xy):
                center_from_mask = self._mask_centroid_from_polygon(masks_xy[index])

            if center_from_mask is None:
                center_u, center_v = self._bbox_center(bbox)
                print(f"[YOLO] Fallback a centroide de bbox en deteccion {index}: mascara no disponible o invalida")
            else:
                center_u, center_v = center_from_mask

            class_id = int(class_ids[index]) if index < len(class_ids) else -1
            detections.append(
                {
                    "bbox": tuple(float(value) for value in bbox[:4]),
                    "center": (float(center_u), float(center_v)),
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": float(confidences[index]) if index < len(confidences) else 0.0,
                }
            )

        return detections

    @staticmethod
    def _overlay_centroid_and_depth(
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        center_key: str = "center",
    ) -> np.ndarray:
        out = frame.copy()
        for detection in detections:
            # Usamos las coordenadas del bbox para saber donde posicionar las letras de las coordenadas de la máscara de segmentación
            x1, y1, x2, y2 = detection["bbox"]
            center = detection.get(center_key)
            if center is None:
                center = detection["center"]
            cx, cy = center
            center_int = (int(round(cx)), int(round(cy)))

            cv2.circle(out, center_int, 5, (0, 0, 255), -1)
            cv2.putText(
                out,
                f"C=({center_int[0]}, {center_int[1]})",
                (int(round(x1)), min(out.shape[0] - 10, int(round(y2)) + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            world_point = detection.get("world")
            if world_point is not None:
                x_world, y_world, z_world = world_point
                depth_text = f"X={x_world:.1f} Y={y_world:.1f} Z={z_world:.1f}"
            else:
                depth_text = "X/Y/Z=inv"

            cv2.putText(
                out,
                depth_text,
                (int(round(x1)), min(out.shape[0] - 10, int(round(y2)) + 38)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        return out

    @staticmethod
    def _yaml() -> Any:
        """Importa PyYAML bajo demanda.

        Se usa un import diferido para no imponer la dependencia si el modulo se
        inspecciona pero no se ejecuta.
        """
        try:
            import yaml

            return yaml
        except ImportError as exc:
            raise RuntimeError("Falta PyYAML. Instala `pyyaml` para leer data.yaml.") from exc

    @staticmethod
    def _yolo_class() -> Any:
        """Importa la clase `YOLO` de Ultralytics bajo demanda.

        Hacemos el import dentro de la funcion porque la dependencia solo hace
        falta en tiempo de ejecucion cuando realmente se entrena o detecta.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            return YOLO
        except ImportError as exc:
            raise RuntimeError("Falta ultralytics. Instala `ultralytics` para entrenar YOLOv8.") from exc

    def _resolve_path(self, value: str | Path | None) -> Optional[Path]:
        """Resuelve una ruta del YAML al sistema de archivos real.

        Roboflow suele exportar rutas relativas como `../train/images`. Esta
        funcion prueba varias posibilidades para convertir esa ruta a una ruta
        utilizable en el workspace actual:

        - primero acepta rutas absolutas que ya existan;
        - despues prueba a interpretarlas relativas al `data.yaml`;
        - por ultimo intenta quitar un `..` inicial si el export lo usa.
        """
        if value is None:
            return None

        candidate = Path(value)
        # Si la ruta ya es absoluta y existe, no hay que transformarla.
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Primera opcion: ruta relativa a la carpeta donde vive el YAML.
        direct = (self.dataset_yaml.parent / candidate).resolve()
        if direct.exists():
            return direct

        # Segunda opcion: algunos exports usan `..` como prefijo redundante.
        if candidate.parts and candidate.parts[0] == "..":
            trimmed = self.dataset_yaml.parent / Path(*candidate.parts[1:])
            if trimmed.exists():
                return trimmed.resolve()

        return direct if direct.exists() else None

    def load_dataset_config(self) -> dict[str, Any]:
        """Lee y valida el `data.yaml` del dataset.

        Devuelve un diccionario con la configuracion cruda y con varios campos
        ya normalizados:

        - `names`: lista de clases ordenada;
        - `nc`: numero de clases detectadas;
        - `train`, `val`, `test`: rutas resueltas a disco cuando existen.

        Si algo importante no cuadra, se lanza una excepcion pronto para no
        descubrir el problema durante el entrenamiento.
        """

        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"No existe el dataset YAML: {self.dataset_yaml}")

        yaml = self._yaml()
        with self.dataset_yaml.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}

        # Roboflow puede guardar `names` como lista o como diccionario indexado.
        names_value = raw_config.get("names")
        if isinstance(names_value, dict):
            names = [str(names_value[key]) for key in sorted(names_value, key=lambda item: int(item))]
        elif isinstance(names_value, list):
            names = [str(name) for name in names_value]
        else:
            raise ValueError("El YAML del dataset no contiene `names` valido.")

        # Convertimos las rutas descritas en el YAML a rutas reales del workspace.
        train_path = self._resolve_path(raw_config.get("train"))
        val_path = self._resolve_path(raw_config.get("val"))
        test_path = self._resolve_path(raw_config.get("test"))

        if train_path is None:
            raise FileNotFoundError("No se pudo resolver la ruta de entrenamiento del dataset.")

        # Si el YAML declara `nc`, comprobamos que coincida con las clases.
        if raw_config.get("nc") is not None and int(raw_config["nc"]) != len(names):
            raise ValueError("`nc` no coincide con el numero de nombres de clase en el YAML.")

        return {
            "raw": raw_config,
            "names": names,
            "nc": len(names),
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }

    def _source_label_path(self, image_path: Path) -> Path:
        """Construye la ruta de la etiqueta asociada a una imagen Roboflow.

        Por convencion, el archivo `foo.jpg` tiene su etiqueta en `labels/foo.txt`.
        """
        return image_path.parent / "labels" / f"{image_path.stem}.txt"

    def _generated_dataset_yaml(self) -> Path:
        """Devuelve la ruta del YAML generado para el split local."""
        return self.dataset_root / ".yolo_split" / "data.yaml"

    def _prepare_dataset_split(self) -> Path:
        """Asegura que exista un conjunto train/valid/test utilizable.

        Si el export de Roboflow ya trae `val` y `test`, no se toca nada y se
        reutiliza el `data.yaml` original.

        Si no hay particiones locales, se crea una carpeta `.yolo_split` con esta
        estructura:

        - `.yolo_split/train/images`
        - `.yolo_split/train/labels`

        - `.yolo_split/valid/images`
        - `.yolo_split/valid/labels`

        - `.yolo_split/test/images`
        - `.yolo_split/test/labels`

        Despues se genera un nuevo `data.yaml` minimo para que Ultralytics pueda
        entrenar sin que tengas que reorganizar el export manualmente.
        """
        config = self.load_dataset_config()

        # Si ya existe validacion y test reales, usamos el YAML original tal cual.
        if config["val"] is not None and config["val"].exists() and config["test"] is not None and config["test"].exists():
            return self.dataset_yaml

        # Si no hay particiones completas, tomamos las imagenes del entrenamiento y las repartimos.
        source_images = config["train"]
        source_labels = source_images.parent / "labels"
        images = sorted(
            [
                path
                for path in source_images.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        )
        if not images:
            raise FileNotFoundError(f"No se han encontrado imagenes en {source_images}")

        # Carpeta interna que solo usamos nosotros para el split local.
        split_root = self.dataset_root / ".yolo_split"
        train_images_dir = split_root / "train" / "images"
        train_labels_dir = split_root / "train" / "labels"

        valid_images_dir = split_root / "valid" / "images"
        valid_labels_dir = split_root / "valid" / "labels"
        
        test_images_dir = split_root / "test" / "images"
        test_labels_dir = split_root / "test" / "labels"

        # Rehacemos el split desde cero para garantizar consistencia.
        if split_root.exists():
            shutil.rmtree(split_root)

        # Creamos la estructura esperada por Ultralytics.
        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)

        valid_images_dir.mkdir(parents=True, exist_ok=True)
        valid_labels_dir.mkdir(parents=True, exist_ok=True)

        test_images_dir.mkdir(parents=True, exist_ok=True)
        test_labels_dir.mkdir(parents=True, exist_ok=True)

        # Barajamos de forma reproducible para que el split sea estable.
        shuffled = images[:]
        random.Random(self.seed).shuffle(shuffled)
        total = len(shuffled)
        if total < 3:
            raise ValueError("Se necesitan al menos 3 imagenes para crear split train/valid/test.")

        train_count = max(1, int(round(total * self.train_fraction)))
        valid_count = max(1, int(round(total * self.val_fraction)))
        test_count = total - train_count - valid_count

        if test_count < 1:
            test_count = 1
            if train_count > valid_count and train_count > 1:
                train_count -= 1
            elif valid_count > 1:
                valid_count -= 1
            else:
                train_count -= 1

        if train_count < 1:
            train_count = 1
        if train_count + valid_count + test_count != total:
            train_count = total - valid_count - test_count

        train_set = set(shuffled[:train_count])
        valid_set = set(shuffled[train_count:train_count + valid_count])
        test_set = set(shuffled[train_count + valid_count:])

        # Copiamos imagen y etiqueta a train, valid o test según corresponda.
        for image_path in shuffled:
            if image_path in train_set:
                target_image_dir = train_images_dir
                target_label_dir = train_labels_dir
            elif image_path in valid_set:
                target_image_dir = valid_images_dir
                target_label_dir = valid_labels_dir
            else:
                target_image_dir = test_images_dir
                target_label_dir = test_labels_dir

            shutil.copy2(image_path, target_image_dir / image_path.name)

            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"No existe la etiqueta asociada: {label_path}")
            shutil.copy2(label_path, target_label_dir / label_path.name)

        # Generamos un YAML nuevo, simple y local, apuntando al split creado.
        generated_yaml = self._generated_dataset_yaml()
        generated_yaml.parent.mkdir(parents=True, exist_ok=True)

        yaml = self._yaml()
        generated_config = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": config["nc"],
            "names": config["names"],
        }

        with generated_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(generated_config, handle, sort_keys=False, allow_unicode=True)

        return generated_yaml

    def load_trained_model(self, weights_path: str | Path | None = None) -> Any:
        """Carga el checkpoint custom entrenado para el modo de deteccion.

        Esta funcion no debe caer en un modelo genérico de COCO. Por eso exige
        un `best.pt` propio, ya sea por parametro o mediante `custom_weights`
        almacenado en la instancia.
        """

        YOLO = self._yolo_class()
        chosen_weights = Path(weights_path) if weights_path is not None else self.custom_weights
        if chosen_weights is None:
            raise FileNotFoundError(
                "No se ha especificado un checkpoint custom entrenado. "
                "Pasa `weights_path` o `custom_weights` con tu best.pt."
            )
        if not chosen_weights.exists():
            raise FileNotFoundError(f"No existe el checkpoint custom: {chosen_weights}")

        # Cargamos el modelo entrenado por nosotros, no un checkpoint genérico.
        self.model = YOLO(str(chosen_weights))
        return self.model

    def train_model(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: Optional[int] = 4,
        device: Optional[str | int] = None,
        patience: int = 20,
        pretrained_weights: str | Path | None = None,
        exist_ok: bool = True,
    ) -> Any:
        """Entrena YOLOv8-seg sobre el dataset de Roboflow.

        Flujo interno:

        1. Asegura que exista un split de validacion utilizable.
        2. Carga un modelo base de YOLOv8-seg.
        3. Lanza `train(...)` con los hiperparametros indicados.
        4. Guarda la salida en `runs/yolo_boxes/...`.

        El resultado final es el objeto cargado con el `best.pt` custom ya
        entrenado. Eso permite pasar directamente de entrenamiento a inferencia
        en la misma ejecucion sin volver a tocar COCO ni pesos base.
        """

        YOLO = self._yolo_class()
        data_yaml = self._prepare_dataset_split()
        model = YOLO(str(pretrained_weights or self.weights))

        # Solo añadimos los argumentos que realmente queremos controlar.
		# patience = nº de epochs consecutivos que el algoritmo debe esperar sin que haya mejoras en mAp antes de detener el entrenamiento automáticamente.
        train_kwargs: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "project": str(self.project_dir),
            "name": self.run_name,
            "patience": patience,
            "exist_ok": exist_ok,
            "verbose": True,
        }
        if batch is not None:
            train_kwargs["batch"] = batch
        if device is not None:
            train_kwargs["device"] = device

        model.train(**train_kwargs)

        best_weights = self.project_dir / self.run_name / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"No se ha generado el checkpoint esperado: {best_weights}")

        # Dejamos listo el modelo custom para inferencia inmediata.
        self.custom_weights = best_weights
        self.model = YOLO(str(best_weights))
        return self.model

    def predict_frame(
        self,
        frame: np.ndarray,
        model: Any | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 512,
    ) -> Tuple[Any, np.ndarray]:
        """Ejecuta inferencia sobre una imagen y devuelve el resultado anotado.

        Args:
            frame: Imagen BGR de entrada.
            model: Modelo YOLO ya cargado. Si no se pasa, se usa `self.model`.
            conf: Umbral minimo de confianza para aceptar detecciones.
            iou: Umbral de solapamiento usado por NMS.
            imgsz: Tamano de inferencia interno para Ultralytics.

        Returns:
            Una tupla con:
            - el objeto `Results` de Ultralytics,
            - la imagen original con cajas, mascaras y texto dibujados.
        """

        # Priorizamos un modelo ya recibido; si no hay, reutilizamos el de la clase.
        active_model = model or self.model
        if active_model is None:
            active_model = self.load_trained_model()

        # Ultralytics devuelve una lista de resultados; usamos el primero porque
        # estamos pasando una sola imagen por llamada.
        results = active_model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        annotated = results[0].plot()
        return results[0], annotated

    def predict_pair_with_depth(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        model: Any | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 512,
    ) -> Tuple[Any, np.ndarray, Any, np.ndarray, list[dict[str, Any]]]:
        """Ejecuta YOLO sobre el par rectificado y añade profundidad en el frame izquierdo.

        El centroide se toma de la máscara de segmentación (`cv2.moments`), con
        fallback al centro del bbox solo si la máscara no está disponible. La
        profundidad y las coordenadas 3D se obtienen a partir de la disparidad
        robusta (mediana de una ventana) calculada con `StereoTriangulator`.
        """
        if self.stereo is None:
            raise RuntimeError("Falta la instancia de StereoTriangulator para calcular profundidad.")

        active_model = model or self.model
        if active_model is None:
            active_model = self.load_trained_model()

        rect_left, rect_right = self.stereo.rectify(frame_left, frame_right)
        disparity = self.stereo.compute_disparity(rect_left, rect_right)

        left_results = active_model.predict(source=rect_left, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        right_results = active_model.predict(source=rect_right, conf=conf, iou=iou, imgsz=imgsz, verbose=False)

        left_detections = self._extract_detections(left_results[0])

        # Solo calculamos las coordenadas del mundo para la imagen izquierda y proyectamos ese mismo punto al lado derecho.
        for detection in left_detections:
            centroid = detection["center"]
            detection["disparity"] = self.stereo.get_disparity_at_centroid(disparity, centroid)
            detection["right_center"] = self.stereo.get_right_center_from_centroid(disparity, centroid)
            detection["world"] = self.stereo.get_3d_from_centroid(disparity, centroid)

        left_annotated = left_results[0].plot()
        right_annotated = right_results[0].plot()

        left_annotated = self._overlay_centroid_and_depth(left_annotated, left_detections)
        right_annotated = self._overlay_centroid_and_depth(right_annotated, left_detections, center_key="right_center")

        return left_results[0], left_annotated, right_results[0], right_annotated, left_detections

    def predict_pair(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        model: Any | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 512,
    ) -> Tuple[Any, np.ndarray, Any, np.ndarray]:
        """Ejecuta YOLO sobre dos frames independientes, uno por lente.

        Esta funcion no hace correspondencia estereo entre ambas vistas. Solo
        corre el detector sobre cada imagen por separado y devuelve las dos
        salidas ya pintadas.
        """

        # Misma logica para la izquierda y la derecha, sin mezclar estereo.
        left_result, left_annotated = self.predict_frame(frame_left, model=model, conf=conf, iou=iou, imgsz=imgsz)
        right_result, right_annotated = self.predict_frame(frame_right, model=model, conf=conf, iou=iou, imgsz=imgsz)
        return left_result, left_annotated, right_result, right_annotated


__all__ = ["YoloSegBoxModule"]
