
# from __future__ import annotations
# import argparse, json, glob, os
# from pathlib import Path
# import numpy as np, torch, cv2, yaml
# from ultralytics import YOLO

# from .adapters import adabn_update
# from .utils import letterbox

# def preprocess_batch(frames_bgr, imgsz, device):
#     import numpy as np, torch
#     xs = []
#     for im in frames_bgr:
#         lb, _, _ = letterbox(im, (imgsz, imgsz), auto=False, scaleup=True)
#         lb = lb[:, :, ::-1]
#         lb = np.ascontiguousarray(lb).astype(np.float32) / 255.0
#         xs.append(lb.transpose(2,0,1))
#     x = torch.from_numpy(np.stack(xs))
#     return x.to(device, non_blocking=True)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--weights", default="outputs/detection/yolov8n_visdrone3/weights/best.pt")
#     ap.add_argument("--data", default="data/visdrone.yaml")
#     ap.add_argument("--imgsz", type=int, default=640)
#     ap.add_argument("--device", default="")
#     ap.add_argument("--adabn_limit", type=int, default=128)
#     ap.add_argument("--out", default="outputs/metrics/tta_val_metrics.json")
#     args = ap.parse_args()

#     yolo = YOLO(args.weights)
#     device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
#     yolo.model.to(device)

#     # try get frames from val path in YAML
#     adapt_frames = []
#     try:
#         with open(args.data, "r") as f:
#             cfg = yaml.safe_load(f)
#         val_path = cfg.get("val") or cfg.get("val_images") or ""
#         if isinstance(val_path, (list, tuple)):
#             val_path = val_path[0]
#         if val_path and os.path.isdir(val_path):
#             paths = sorted(glob.glob(os.path.join(val_path, "*.jpg")))[:args.adabn_limit]
#             for p in paths:
#                 im = cv2.imread(p)
#                 if im is not None:
#                     adapt_frames.append(im)
#     except Exception:
#         pass

#     if len(adapt_frames):
#         bs = 8
#         batches = [adapt_frames[i:i+bs] for i in range(0, len(adapt_frames), bs)]
#         def gen():
#             for b in batches:
#                 yield preprocess_batch(b, args.imgsz, device)
#         n = adabn_update(yolo.model, gen(), device=device, max_batches=len(batches))
#         print(f"[adabn] updated BN on {n} mini-batches")
#     else:
#         print("[adabn] no frames found; skipping BN update")

#     metrics = yolo.val(data=args.data, split="val", imgsz=args.imgsz, device=device,
#                        conf=0.001, iou=0.6, augment=True)
#     payload = {
#         "map50_95": metrics.box.map,
#         "map50": metrics.box.map50,
#         "maps_per_class": list(metrics.box.maps),
#         "speed_ms": dict(metrics.speed),
#         "tta": {"augment": True, "adabn": bool(len(adapt_frames))}
#     }
#     Path(args.out).parent.mkdir(parents=True, exist_ok=True)
#     with open(args.out, "w") as f:
#         json.dump(payload, f, indent=2)
#     print(json.dumps(payload, indent=2))

# if __name__ == "__main__":
#     main()

#-------------------------------------------------

# run_tta_eval_visdrone.py

# from __future__ import annotations
 
# import argparse

# import json

# import random

# from pathlib import Path

# from typing import List
 
# import cv2

# import numpy as np

# import torch

# import yaml

# from ultralytics import YOLO
 
# from adapters import adabn_update, tent_adapt, pseudo_label_adapt
 
 
# IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
 
 
# def parse_args() -> argparse.Namespace:

#     p = argparse.ArgumentParser(

#         description="Evaluate YOLOv8 on VisDrone with optional test-time adaptation."

#     )

#     p.add_argument(

#         "--weights",

#         type=str,

#         required=True,

#         help="Path to trained YOLO weights (e.g. best.pt).",

#     )

#     p.add_argument(

#         "--data",

#         type=str,

#         required=True,

#         help="Path to data YAML (e.g. data/visdrone.yaml).",

#     )

#     p.add_argument("--imgsz", type=int, default=640)

#     p.add_argument(

#         "--device",

#         type=str,

#         default="cuda",

#         help="Device for Ultralytics (e.g. 'cuda', 'cpu', '0').",

#     )

#     p.add_argument(

#         "--tta-mode",

#         choices=["none", "adabn", "tent", "pseudo"],

#         default="adabn",

#         help="TTA variant to apply before evaluation.",

#     )

#     p.add_argument(

#         "--tta-batches",

#         type=int,

#         default=64,

#         help="Max adaptation batches (each batch is a single image here).",

#     )

#     p.add_argument(

#         "--tta-steps",

#         type=int,

#         default=1,

#         help="Inner gradient steps per batch (Tent / pseudo).",

#     )

#     p.add_argument(

#         "--tta-lr",

#         type=float,

#         default=1e-4,

#         help="LR for Tent / pseudo-label adaptation.",

#     )

#     p.add_argument(

#         "--tta-conf-thr",

#         type=float,

#         default=0.7,

#         help="Confidence threshold for pseudo-label adaptation.",

#     )

#     p.add_argument(

#         "--out",

#         type=str,

#         default="outputs/metrics/tta_val_metrics.json",

#         help="Where to save metrics JSON.",

#     )

#     return p.parse_args()
 
 
# def load_val_images_from_yaml(

#     data_yaml: str, limit: int | None = None

# ) -> List[Path]:

#     """

#     Load image paths from the validation split specified in the YAML.

#     """

#     with open(data_yaml, "r") as f:

#         cfg = yaml.safe_load(f)
 
#     root = Path(cfg.get("path", "")).expanduser()

#     val_entry = cfg.get("val")
 
#     if val_entry is None:

#         raise ValueError("YAML must specify a 'val' entry for the validation split.")
 
#     val_path = (root / val_entry) if root and not Path(val_entry).is_absolute() else Path(val_entry)

#     if not val_path.exists():

#         raise FileNotFoundError(f"Validation path not found: {val_path}")
 
#     imgs = [

#         p

#         for p in val_path.rglob("*")

#         if p.suffix.lower() in IMG_EXTS and p.is_file()

#     ]

#     imgs.sort()

#     if limit is not None:

#         imgs = imgs[:limit]

#     return imgs
 
 
# def preprocess_images(

#     image_paths: List[Path], imgsz: int, device: torch.device

# ) -> List[torch.Tensor]:

#     """

#     Read images with OpenCV, resize to imgsz x imgsz, and convert to BCHW float tensors.

#     Each tensor is [1, 3, H, W].

#     """

#     batches: List[torch.Tensor] = []

#     for p in image_paths:

#         im = cv2.imread(str(p))

#         if im is None:

#             continue

#         # BGR -> RGB

#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#         im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

#         im = im.astype(np.float32) / 255.0

#         im = np.transpose(im, (2, 0, 1))  # HWC -> CHW

#         t = torch.from_numpy(im).unsqueeze(0).to(device, non_blocking=True)

#         batches.append(t)

#     return batches
 
 
# def main() -> None:

#     args = parse_args()

#     device = torch.device(args.device if args.device != "auto" else "cuda")
 
#     # Load model

#     yolo = YOLO(args.weights)

#     yolo.to(device)  # Ultralytics convenience
 
#     # ------------------------------------------------------------------

#     # 1) Sample adaptation frames from val set

#     # ------------------------------------------------------------------

#     if args.tta_mode != "none":

#         # We treat each image as batch size 1, so max_batches == max images

#         img_paths = load_val_images_from_yaml(args.data, limit=args.tta_batches)

#         if not img_paths:

#             print("No validation images found for adaptation; skipping TTA.")

#             n_adapt = 0

#         else:

#             adap_batches = preprocess_images(img_paths, args.imgsz, device)
 
#             if args.tta_mode == "adabn":

#                 n_adapt = adabn_update(

#                     yolo.model, adap_batches, device=device, max_batches=args.tta_batches

#                 )

#             elif args.tta_mode == "tent":

#                 n_adapt = tent_adapt(

#                     yolo.model,

#                     adap_batches,

#                     device=device,

#                     max_batches=args.tta_batches,

#                     steps_per_batch=args.tta_steps,

#                     lr=args.tta_lr,

#                 )

#             else:  # pseudo

#                 n_adapt = pseudo_label_adapt(

#                     yolo.model,

#                     adap_batches,

#                     device=device,

#                     max_batches=args.tta_batches,

#                     steps_per_batch=args.tta_steps,

#                     lr=args.tta_lr,

#                     conf_thr=args.tta_conf_thr,

#                 )

#             print(f"TTA mode={args.tta_mode}, adapted on {n_adapt} mini-batches.")

#     else:

#         n_adapt = 0

#         print("TTA mode=none, skipping adaptation, evaluating baseline model.")
 
#     # ------------------------------------------------------------------

#     # 2) Run validation

#     # ------------------------------------------------------------------

#     metrics = yolo.val(

#         data=args.data,

#         imgsz=args.imgsz,

#         device=str(device).replace("cuda:", "cuda"),

#         split="val",

#     )
 
#     out_path = Path(args.out)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
 
#     payload = {

#         "map50_95": float(metrics.box.map),

#         "map50": float(metrics.box.map50),

#         "maps_per_class": [float(x) for x in metrics.box.maps.tolist()],

#         "speed_ms": {k: float(v) for k, v in metrics.speed.items()},

#         "tta": {

#             "mode": args.tta_mode,

#             "n_adapt_batches": int(n_adapt),

#             "imgsz": args.imgsz,

#         },

#     }
 
#     with open(out_path, "w") as f:

#         json.dump(payload, f, indent=2)
 
#     print(f"TTA evaluation complete. Metrics saved to {out_path.resolve()}")
 
 
# if __name__ == "__main__":

#     main()

#----------------speed------------------
 
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

from adapters import adabn_update, tent_adapt, pseudo_label_adapt


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate YOLOv8 on VisDrone with optional test-time adaptation."
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
        help="Path to data YAML (e.g. data/visdrone.yaml).",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Ultralytics (e.g. 'cuda', 'cpu', '0').",
    )
    p.add_argument(
        "--tta-mode",
        choices=["none", "adabn", "tent", "pseudo"],
        default="adabn",
        help="TTA variant to apply before evaluation.",
    )
    p.add_argument(
        "--tta-batches",
        type=int,
        default=64,
        help="Max adaptation batches (each batch is a single image here).",
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
        default="outputs/metrics/tta_val_metrics.json",
        help="Where to save metrics JSON.",
    )
    return p.parse_args()


def load_val_images_from_yaml(
    data_yaml: str, limit: int | None = None
) -> List[Path]:
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "")).expanduser()
    val_entry = cfg.get("val")
    if val_entry is None:
        raise ValueError("YAML must specify a 'val' entry for the validation split.")

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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else "cuda")

    yolo = YOLO(args.weights)
    yolo.to(device)

    if args.tta_mode != "none":
        img_paths = load_val_images_from_yaml(args.data, limit=args.tta_batches)
        if not img_paths:
            print("No validation images found for adaptation; skipping TTA.")
            n_adapt = 0
        else:
            adap_batches = preprocess_images(img_paths, args.imgsz, device)

            if args.tta_mode == "adabn":
                n_adapt = adabn_update(
                    yolo.model, adap_batches, device=device, max_batches=args.tta_batches
                )
            elif args.tta_mode == "tent":
                n_adapt = tent_adapt(
                    yolo.model,
                    adap_batches,
                    device=device,
                    max_batches=args.tta_batches,
                    steps_per_batch=args.tta_steps,
                    lr=args.tta_lr,
                )
            else:
                n_adapt = pseudo_label_adapt(
                    yolo.model,
                    adap_batches,
                    device=device,
                    max_batches=args.tta_batches,
                    steps_per_batch=args.tta_steps,
                    lr=args.tta_lr,
                    conf_thr=args.tta_conf_thr,
                )
            print(f"TTA mode={args.tta_mode}, adapted on {n_adapt} mini-batches.")
    else:
        n_adapt = 0
        print("TTA mode=none, skipping adaptation, evaluating baseline model.")

    metrics = yolo.val(
        data=args.data,
        imgsz=args.imgsz,
        device=str(device).replace("cuda:", "cuda"),
        split="val",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "maps_per_class": [float(x) for x in metrics.box.maps.tolist()],
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        "tta": {
            "mode": args.tta_mode,
            "n_adapt_batches": int(n_adapt),
            "imgsz": args.imgsz,
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"TTA evaluation complete. Metrics saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
