"""Entrenamiento standalone de YOLOv8-seg para el dataset de cajas.

Este script permite lanzar el entrenamiento desde un portatil sin abrir la
cámara ZED ni ejecutar la aplicación interactiva principal.
"""

from __future__ import annotations

import argparse
from importlib import util as importlib_util
from pathlib import Path


def load_yolo_module_class() -> type:
    module_path = Path(__file__).resolve().parent / "modules" / "yolo_stereo_module.py"
    spec = importlib_util.spec_from_file_location("yolo_stereo_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el modulo desde {module_path}")

    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YoloSegBoxModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena YOLOv8-seg sobre el dataset etiquetado_5_tipos_de_cajas.yolov8."
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("etiquetado_5_tipos_de_cajas.yolov8") / "data.yaml",
        help="Ruta al data.yaml del dataset de Roboflow.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n-seg.pt",
        help="Pesos base de Ultralytics para arrancar el entrenamiento.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("runs") / "yolo_boxes",
        help="Carpeta donde se guardaran los runs de entrenamiento.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="boxes_seg",
        help="Nombre del experimento dentro de project-dir.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Numero de epocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen para entrenamiento.")
    parser.add_argument("--batch", type=int, default=4, help="Tamano de batch recomendado para empezar.")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo de entrenamiento, por ejemplo 0, cpu o cuda:0.")
    parser.add_argument("--patience", type=int, default=20, help="Patience para early stopping.")
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Ruta opcional a unos pesos custom para reanudar o afinar.",
    )
    parser.add_argument(
        "--no-exist-ok",
        action="store_true",
        help="Fuerza a no reutilizar runs previos con el mismo nombre.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"No existe el dataset YAML: {args.dataset_yaml}")

    YoloSegBoxModule = load_yolo_module_class()

    trainer = YoloSegBoxModule(
        dataset_yaml=args.dataset_yaml,
        weights=args.weights,
        project_dir=args.project_dir,
        run_name=args.run_name,
    )

    print(f"Dataset: {args.dataset_yaml}")
    print(f"Proyecto: {args.project_dir}")
    print(f"Run: {args.run_name}")
    print("Iniciando entrenamiento YOLOv8-seg...")

    model = trainer.train_model(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        pretrained_weights=args.pretrained_weights,
        exist_ok=not args.no_exist_ok,
    )

    print("Entrenamiento terminado correctamente.")
    print(f"Checkpoint listo en: {trainer.custom_weights}")
    print(f"Modelo cargado: {model}")


if __name__ == "__main__":
    main()