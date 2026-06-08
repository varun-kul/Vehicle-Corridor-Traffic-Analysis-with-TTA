#!/usr/bin/env python3
"""
eval_baseline_vs_tta_detection.py
 
Compare YOLOv8 detection metrics with and without TTA on VisDrone.
 
Usage example (PowerShell):
  python eval_baseline_vs_tta_detection.py `
    --weights outputs/detection/yolov8n_visdrone3/weights/best.pt `
    --data data/processed/yolo/visdrone.yaml `
    --imgsz 640 `
    --device cuda `
    --tta-mode adabn `
    --tta-batches 128 `
    --out outputs/metrics/baseline_vs_tta_visdrone.json
"""
 
from __future__ import annotations
 
import argparse
import json
from pathlib import Path
from typing import List
 
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
 
from TTA.adapters import adabn_update, tent_adapt, pseudo_label_adapt
 
 
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate baseline vs TTA detection metrics on VisDrone."
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO weights (e.g. best.pt).",
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data YAML (e.g. data/processed/yolo/visdrone.yaml).",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string for Ultralytics (e.g. 'cuda', 'cpu', '0').",
    )
    p.add_argument(
        "--tta-mode",
        choices=["adabn", "tent", "pseudo"],
        default="adabn",
        help="Which TTA variant to evaluate against baseline.",
    )
    p.add_argument(
        "--tta-batches",
        type=int,
        default=64,
        help="Number of mini-batches (1 image each) used for adaptation.",
    )
    p.add_argument(
        "--tta-steps",
        type=int,
        default=1,
        help="Inner gradient steps per batch (Tent / pseudo).",
    )
    p.add_argument(
        "--tta-lr",
        type=float,
        default=1e-4,
        help="LR for Tent / pseudo-label adaptation.",
    )
    p.add_argument(
        "--tta-conf-thr",
        type=float,
        default=0.7,
        help="Confidence threshold for pseudo-label adaptation.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs/metrics/baseline_vs_tta_visdrone.json",
        help="Where to save metrics comparison JSON.",
    )
    return p.parse_args()
 
 
def load_val_images_from_yaml(
    data_yaml: str, limit: int | None = None
) -> list[Path]:
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
 
    root = Path(cfg.get("path", "")).expanduser()
    val_entry = cfg.get("val")
    if val_entry is None:
        raise ValueError("YAML must define a 'val' split.")
 
    val_path = (root / val_entry) if root and not Path(val_entry).is_absolute() else Path(val_entry)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation path not found: {val_path}")
 
    imgs = [
        p
        for p in val_path.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]
    imgs.sort()
    if limit is not None:
        imgs = imgs[:limit]
    return imgs
 
 
def preprocess_images(
    image_paths: List[Path], imgsz: int, device: torch.device
) -> List[torch.Tensor]:
    batches: List[torch.Tensor] = []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))
        t = torch.from_numpy(im).unsqueeze(0).to(device, non_blocking=True)
        batches.append(t)
    return batches
 
 
def metrics_to_dict(metrics) -> dict:
    """
    Extract key metrics from Ultralytics val() result.
 
    `.box.map`      -> mAP50-95
    `.box.map50`    -> mAP50
    `.box.mp`       -> mean precision
    `.box.mr`       -> mean recall
    """
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "mp": float(metrics.box.mp),
        "mr": float(metrics.box.mr),
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }
 
 
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else "cuda")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    print(f"Loading YOLO model from {args.weights} ...")
    yolo = YOLO(args.weights)
    yolo.to(device)
 
    # ------------------- Baseline (no TTA) -------------------
    print("\n[1] Running baseline validation (no TTA) ...")
    baseline_metrics = yolo.val(
        data=args.data,
        imgsz=args.imgsz,
        device=str(device).replace("cuda:", "cuda"),
        split="val",
    )
    baseline_dict = metrics_to_dict(baseline_metrics)
    print(
        f"Baseline: mAP50-95={baseline_dict['map50_95']:.4f}, "
        f"mAP50={baseline_dict['map50']:.4f}, "
        f"mp={baseline_dict['mp']:.4f}, mr={baseline_dict['mr']:.4f}"
    )
 
    # ------------------- TTA adaptation ----------------------
    print(f"\n[2] Adapting model with TTA mode='{args.tta_mode}' ...")
    img_paths = load_val_images_from_yaml(args.data, limit=args.tta_batches)
    if not img_paths:
        print("No validation images found for adaptation; TTA step skipped.")
        n_adapt = 0
    else:
        adap_batches = preprocess_images(img_paths, args.imgsz, device)
        if args.tta_mode == "adabn":
            n_adapt = adabn_update(
                yolo.model,
                adap_batches,
                device=device,
                max_batches=len(adap_batches),
            )
        elif args.tta_mode == "tent":
            n_adapt = tent_adapt(
                yolo.model,
                adap_batches,
                device=device,
                max_batches=len(adap_batches),
                steps_per_batch=args.tta_steps,
                lr=args.tta_lr,
            )
        else:
            n_adapt = pseudo_label_adapt(
                yolo.model,
                adap_batches,
                device=device,
                max_batches=len(adap_batches),
                steps_per_batch=args.tta_steps,
                lr=args.tta_lr,
                conf_thr=args.tta_conf_thr,
            )
    print(f"TTA adaptation done on {n_adapt} mini-batches.")
 
    # ------------------- Validation WITH TTA -----------------
    print("\n[3] Running validation with adapted model ...")
    tta_metrics = yolo.val(
        data=args.data,
        imgsz=args.imgsz,
        device=str(device).replace("cuda:", "cuda"),
        split="val",
    )
    tta_dict = metrics_to_dict(tta_metrics)
    print(
        f"TTA-{args.tta_mode}: mAP50-95={tta_dict['map50_95']:.4f}, "
        f"mAP50={tta_dict['map50']:.4f}, "
        f"mp={tta_dict['mp']:.4f}, mr={tta_dict['mr']:.4f}"
    )
 
    # ------------------- Pretty diff print -------------------
    print("\n[4] Comparison (TTA - Baseline):")
    def diff(k: str) -> float:
        return tta_dict[k] - baseline_dict[k]
 
    print(f"Δ mAP50-95: {diff('map50_95'):+.4f}")
    print(f"Δ mAP50:    {diff('map50'):+.4f}")
    print(f"Δ mp:       {diff('mp'):+.4f}")
    print(f"Δ mr:       {diff('mr'):+.4f}")
 
    # ------------------- Save JSON ---------------------------
    payload = {
        "baseline": baseline_dict,
        "tta": {
            "mode": args.tta_mode,
            "n_adapt_batches": int(n_adapt),
            "metrics": tta_dict,
        },
        "imgsz": args.imgsz,
    }
 
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
 
    print(f"\nMetrics comparison saved to {out_path.resolve()}")
 
 
if __name__ == "__main__":
    main()