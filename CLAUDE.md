# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Stereo 3D vision system for bin picking using a ZED2 stereo camera. The pipeline detects boxes (5 types: busbar, chasis, manguera, pcb, potencia) with YOLOv8-seg, computes their 3D centroids via stereo disparity, and displays results in an interactive OpenCV window. There is also a separate PyBullet simulation sandbox (`prueba_bullet.py`).

## Running the app

```bash
# Main interactive app (requires ZED2 camera and trained best.pt)
python main.py

# Train YOLOv8-seg without opening the camera
python train_yolo_dataset.py
python train_yolo_dataset.py --epochs 50 --batch 8 --device 0

# PyBullet physics sandbox
python prueba_bullet.py
```

The app starts in YOLO mode (mode 5) automatically. Keys: `1` rectify | `2` triangulate (click left then right) | `3` disparity | `4` gestures | `5` YOLO | `r` clear points | `q` quit.

## Key dependencies

- `pyzed` (ZED SDK): install from the bundled wheel `pyzed-5.3-cp312-cp312-linux_x86_64.whl`
- `ultralytics`: YOLOv8 training and inference
- `mediapipe`: gesture recognition (mode 4, downloaded on first use)
- `pybullet`, `pybullet_data`: physics simulation
- `opencv-python`, `numpy`, `pyyaml`

## Architecture

```
main.py                        # StereoInteractiveApp — mode dispatcher and main loop
modules/
  camera_module.py             # ZEDCamera — thin wrapper around pyzed.sl
  stereo_module.py             # StereoTriangulator — calibration load, rectify, SGBM disparity, triangulation, Q-matrix 3D
  yolo_stereo_module.py        # YoloSegBoxModule — dataset prep, train, predict_frame/predict_pair_with_depth
  gesture_module.py            # GestureRecognizer — MediaPipe gesture recognition
  prelabel_capture_module.py   # PrelabelCaptureModule — saves ZED frames for Roboflow labeling
utils/
  visualization.py             # draw_points_and_text helper
calibration/
  stereo_calib.npz             # Required: Kl, Dl, Kr, Dr, R, T arrays
config/
  box_types.yaml               # Physical box dimensions (Wallbox only, kept for reference)
etiquetado_5_tipos_de_cajas.yolov8/
  data.yaml                    # Roboflow export: 5 classes, train/val/test split paths
runs/yolo_boxes/boxes_seg/weights/best.pt  # Trained checkpoint (expected location)
```

## Data flow in mode 5 (YOLO + depth)

1. `ZEDCamera.get_frames()` → BGR left/right frames
2. `StereoTriangulator.rectify()` → undistorted+aligned pair
3. `StereoTriangulator.compute_disparity()` → SGBM disparity map
4. `YoloSegBoxModule` runs YOLOv8-seg on both rectified frames independently
5. For each detection in the left frame: centroid from mask polygon (fallback: bbox center) → `get_disparity_at_centroid()` → `get_3d_from_centroid()` (via `cv2.reprojectImageTo3D` with Q matrix) → world X/Y/Z in mm
6. Right frame centroids are projected from left using disparity offset: `right_u = left_u - disparity`

## Calibration format

`stereo_calib.npz` must contain keys: `Kl`, `Dl`, `Kr`, `Dr`, `R`, `T`. The triangulation units (mm) match the units of `T`. `StereoTriangulator` is initialized with `image_size=(1280, 720)` (HD720).

## Dataset and training

The Roboflow export at `etiquetado_5_tipos_de_cajas.yolov8/` uses relative paths (`../train/images`). If `val/` or `test/` splits are missing, `YoloSegBoxModule._prepare_dataset_split()` auto-generates a `.yolo_split/` folder with a 70/20/10 split. The generated `data.yaml` is at `.yolo_split/data.yaml`. To capture new training images from the ZED, use `PrelabelCaptureModule.capture()` (saves to `images_pre_labeled_3/` by default).

## Important constraints

- `pyzed` uses `sl.DEPTH_MODE.NONE` — depth comes from our own SGBM pipeline, not ZED's built-in depth.
- `YoloSegBoxModule` does lazy imports of `ultralytics` and `yaml` to avoid hard failures when those packages aren't present.
- `load_trained_model()` refuses to fall back to COCO weights — it requires an explicit `best.pt` path.
- SGBM `numDisparities=160` (16×10); changing this affects the depth range and must remain a multiple of 16.
