"""
train_burrow.py
===============
Fine-tune YOLOv8-nano on the Lysiosquillina burrow dataset.

Usage:
    python train_burrow.py --data dataset_config.yaml
    python train_burrow.py --data dataset_config.yaml --epochs 200 --batch 16
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed.  Run:  pip install ultralytics")


AUGMENTATION_HYPERPARAMS = {
    "fliplr":     0.5,
    "flipud":     0.1,
    "degrees":    10.0,
    "translate":  0.1,
    "scale":      0.3,
    "shear":      2.0,
    "hsv_h":      0.015,
    "hsv_s":      0.4,
    "hsv_v":      0.3,
    "blur":       0.01,
    "mosaic":     1.0,
    "mixup":      0.1,
    "copy_paste": 0.05,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8-nano on burrow dataset")
    p.add_argument("--data",     required=True, help="Path to dataset_config.yaml")
    p.add_argument("--weights",  default="yolov8n.pt")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--project",  default="runs/burrow_train")
    p.add_argument("--name",     default=None)
    p.add_argument("--device",   default="")
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--no-export-onnx", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    run_name = args.name or f"burrow_{int(time.time())}"
    print(f"\n{'='*60}")
    print(f"  Archimedes Survey -- Burrow Detection Training")
    print(f"  Weights : {args.weights}")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs}  (patience={args.patience})")
    print(f"  Batch   : {args.batch}   ImgSz={args.imgsz}")
    print(f"  Run     : {args.project}/{run_name}")
    print(f"{'='*60}\n")

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=run_name,
        device=args.device if args.device else None,
        workers=args.workers,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        label_smoothing=0.05,
        **AUGMENTATION_HYPERPARAMS,
    )

    print("\n" + "="*60)
    print("  Training complete -- metrics summary")
    print("="*60)
    try:
        metrics = model.val()
        print(f"  mAP@0.50      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.50:0.95 : {metrics.box.map:.4f}")
        print(f"  Precision     : {metrics.box.mp:.4f}")
        print(f"  Recall        : {metrics.box.mr:.4f}")
    except Exception as exc:
        print(f"  (Could not retrieve val metrics: {exc})")

    best_pt = Path(args.project) / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(args.project) / run_name / "weights" / "last.pt"

    print(f"\n  Best model : {best_pt}")
    dest_pt = Path(args.project) / "burrow_best.pt"
    shutil.copy2(best_pt, dest_pt)
    print(f"  Copied     : {dest_pt}")

    if not args.no_export_onnx:
        print("\n  Exporting to ONNX ...")
        try:
            best_model = YOLO(str(best_pt))
            onnx_path  = best_model.export(
                format="onnx", imgsz=args.imgsz,
                simplify=True, opset=12, half=False, dynamic=False,
            )
            print(f"  ONNX model : {onnx_path}")
        except Exception as exc:
            print(f"  ONNX export failed: {exc}")
            print("  Install:  pip install onnx onnxruntime")
    else:
        print("  (ONNX export skipped)")

    print("\n  Done.\n")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
