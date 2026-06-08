
# from __future__ import annotations
# import argparse, csv
# from pathlib import Path
# import numpy as np, cv2, torch
# from ultralytics import YOLO

# from .utils import frames_from_source, letterbox
# from .adapters import adabn_update

# def preprocess_batch(frames_bgr, imgsz, device):
#     import numpy as np, torch
#     xs = []
#     for im in frames_bgr:
#         lb, _, _ = letterbox(im, (imgsz, imgsz), auto=False, scaleup=True)
#         lb = lb[:, :, ::-1]  # BGR->RGB
#         lb = np.ascontiguousarray(lb).astype(np.float32) / 255.0
#         xs.append(lb.transpose(2,0,1))  # C,H,W
#     x = torch.from_numpy(np.stack(xs))
#     return x.to(device, non_blocking=True)

# def track_stream_with_tta(source, weights, imgsz=640, device="", conf=0.25, iou=0.5,
#                           adabn_frames=64, frame_stride=1, augment_infer=True,
#                           save_video=True, out_video="outputs/tta/annotated.mp4",
#                           out_csv="outputs/tta/tracks.csv", tracker_config="bytetrack.yaml"):

#     Path(out_video).parent.mkdir(parents=True, exist_ok=True)
#     Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

#     print(f"[load] weights: {weights}")
#     yolo = YOLO(weights)
#     mdl = yolo.model

#     device_resolved = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#     mdl.to(device_resolved)

#     # 1) AdaBN on unlabeled target frames
#     print(f"[adabn] collecting {adabn_frames} frames for BN adaptation ...")
#     adapt_frames = [f for f in frames_from_source(source, every_n=frame_stride, limit=adabn_frames)]
#     if len(adapt_frames) > 0:
#         bs = 8
#         batches = [adapt_frames[i:i+bs] for i in range(0, len(adapt_frames), bs)]
#         def gen():
#             for b in batches:
#                 yield preprocess_batch(b, imgsz, device_resolved)
#         n = adabn_update(mdl, gen(), device=device_resolved, max_batches=len(batches))
#         print(f"[adabn] updated BN on {n} batches ({len(adapt_frames)} frames)")
#     else:
#         print("[adabn] no frames gathered; skipping BN update")

#     # 2) Tracking with Ultralytics stream API
#     print(f"[track] tracker='{tracker_config}', augment_infer={augment_infer}")
#     stream = yolo.track(source=source, imgsz=imgsz, conf=conf, iou=iou, device=device_resolved,
#                         tracker=tracker_config, persist=True, stream=True, verbose=True, augment=augment_infer)

#     vw = None
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     with open(out_csv, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["frame","track_id","cls","conf","x1","y1","x2","y2"])
#         frame_idx = -1
#         for r in stream:
#             frame_idx += 1
#             frame = r.orig_img
#             if save_video and vw is None:
#                 h, w = frame.shape[:2]
#                 vw = cv2.VideoWriter(out_video, fourcc, 25, (w, h))

#             if r.boxes is None or r.boxes.xyxy.numel() == 0:
#                 if save_video and vw: vw.write(frame)
#                 continue

#             xyxy = r.boxes.xyxy.cpu().numpy()
#             confs = r.boxes.conf.cpu().numpy()
#             clses = r.boxes.cls.cpu().numpy().astype(int)
#             ids = r.boxes.id
#             ids = ids.cpu().numpy().astype(int) if ids is not None else np.arange(len(xyxy))

#             for (x1,y1,x2,y2), c, k, tid in zip(xyxy, confs, clses, ids):
#                 writer.writerow([frame_idx, int(tid), int(k), float(c), float(x1), float(y1), float(x2), float(y2)])
#                 x1i,y1i,x2i,y2i = map(int, [x1,y1,x2,y2])
#                 cv2.rectangle(frame,(x1i,y1i),(x2i,y2i),(0,255,0),2)
#                 cv2.putText(frame, f"ID:{tid} c:{c:.2f}", (x1i, max(0,y1i-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#             if save_video and vw:
#                 vw.write(frame)

#     if vw: vw.release()
#     print(f"[done] wrote: {out_video}")
#     print(f"[done] wrote: {out_csv}")

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--source", required=True, help="video path, camera index, or directory of images")
#     ap.add_argument("--weights", default="outputs/detection/yolov8n_visdrone3/weights/best.pt")
#     ap.add_argument("--imgsz", type=int, default=640)
#     ap.add_argument("--device", default="")
#     ap.add_argument("--conf", type=float, default=0.25)
#     ap.add_argument("--iou", type=float, default=0.5)
#     ap.add_argument("--adabn_frames", type=int, default=64)
#     ap.add_argument("--frame_stride", type=int, default=1)
#     ap.add_argument("--no_augment", action="store_true")
#     ap.add_argument("--save_video", action="store_true")
#     ap.add_argument("--out_video", default="outputs/tta/annotated.mp4")
#     ap.add_argument("--out_csv", default="outputs/tta/tracks.csv")
#     ap.add_argument("--tracker", default="bytetrack.yaml")
#     args = ap.parse_args()

#     track_stream_with_tta(
#         source=args.source,
#         weights=args.weights,
#         imgsz=args.imgsz,
#         device=args.device,
#         conf=args.conf,
#         iou=args.iou,
#         adabn_frames=args.adabn_frames,
#         frame_stride=args.frame_stride,
#         augment_infer=not args.no_augment,
#         save_video=args.save_video,
#         out_video=args.out_video,
#         out_csv=args.out_csv,
#         tracker_config=args.tracker,
#     )

# if __name__ == "__main__":
#     main()


#----------------------------------------

# run_tta_track.py

# from __future__ import annotations
 
# import argparse

# import csv

# from pathlib import Path

# from typing import List, Tuple
 
# import cv2

# import numpy as np

# import torch

# from ultralytics import YOLO
 
# from adapters import adabn_update, tent_adapt, pseudo_label_adapt
 
 
# def parse_args() -> argparse.Namespace:

#     p = argparse.ArgumentParser(

#         description="Run YOLOv8+ByteTrack on a video with optional TTA."

#     )

#     p.add_argument(

#         "--source",

#         type=str,

#         required=True,

#         help="Path to input video file (e.g. MOT17-02-FRCNN.mp4).",

#     )

#     p.add_argument(

#         "--weights",

#         type=str,

#         required=True,

#         help="Path to trained YOLO weights (e.g. best.pt).",

#     )

#     p.add_argument("--imgsz", type=int, default=640)

#     p.add_argument(

#         "--device",

#         type=str,

#         default="cuda",

#         help="Device for Ultralytics (e.g. 'cuda', 'cpu', '0').",

#     )

#     p.add_argument("--conf", type=float, default=0.25)

#     p.add_argument("--iou", type=float, default=0.5)

#     p.add_argument(

#         "--tracker",

#         type=str,

#         default="bytetrack.yaml",

#         help="Tracker config for YOLO.track (default ByteTrack).",

#     )

#     p.add_argument(

#         "--tta-mode",

#         choices=["none", "adabn", "tent", "pseudo"],

#         default="adabn",

#         help="TTA variant to apply before tracking.",

#     )

#     p.add_argument(

#         "--tta-frames",

#         type=int,

#         default=64,

#         help="Number of frames from the video to use for adaptation.",

#     )

#     p.add_argument(

#         "--tta-step",

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

#         "--out-video",

#         type=str,

#         default="outputs/tta/tta_track.mp4",

#         help="Path to save annotated output video.",

#     )

#     p.add_argument(

#         "--out-csv",

#         type=str,

#         default="outputs/tta/tta_tracks.csv",

#         help="Path to save tracks CSV.",

#     )

#     return p.parse_args()
 
 
# def sample_frames_for_tta(

#     video_path: str, imgsz: int, n_frames: int, device: torch.device

# ) -> List[torch.Tensor]:

#     """

#     Sample up to n_frames roughly evenly spaced frames from a video,

#     resize to imgsz x imgsz, and convert to tensors [1,3,H,W].

#     """

#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():

#         raise RuntimeError(f"Could not open video: {video_path}")
 
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if frame_count <= 0:

#         frame_count = n_frames
 
#     idxs = np.linspace(0, frame_count - 1, num=min(n_frames, frame_count), dtype=int)

#     idx_set = set(int(i) for i in idxs.tolist())
 
#     batches: List[torch.Tensor] = []

#     cur = 0

#     while True:

#         ret, frame = cap.read()

#         if not ret:

#             break

#         if cur in idx_set:

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

#             rgb = rgb.astype(np.float32) / 255.0

#             rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW

#             t = torch.from_numpy(rgb).unsqueeze(0).to(device, non_blocking=True)

#             batches.append(t)

#             if len(batches) >= n_frames:

#                 break

#         cur += 1
 
#     cap.release()

#     return batches
 
 
# def get_video_writer(

#     out_path: Path, frame_size: Tuple[int, int], fps: float

# ) -> cv2.VideoWriter:

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

#     if not writer.isOpened():

#         raise RuntimeError(f"Could not open VideoWriter for {out_path}")

#     return writer
 
 
# def main() -> None:

#     args = parse_args()

#     device = torch.device(args.device if args.device != "auto" else "cuda")
 
#     src_path = Path(args.source)

#     if not src_path.exists():

#         raise FileNotFoundError(f"Source video not found: {src_path}")
 
#     # Load model

#     yolo = YOLO(args.weights)

#     yolo.to(device)
 
#     # ------------------------------------------------------------------

#     # 1) TTA adaptation on sampled frames

#     # ------------------------------------------------------------------

#     if args.tta_mode != "none":

#         adap_batches = sample_frames_for_tta(

#             str(src_path), args.imgsz, args.tta_frames, device

#         )

#         if not adap_batches:

#             print("No frames collected for TTA; skipping adaptation.")

#             n_adapt = 0

#         else:

#             if args.tta_mode == "adabn":

#                 n_adapt = adabn_update(

#                     yolo.model, adap_batches, device=device, max_batches=len(adap_batches)

#                 )

#             elif args.tta_mode == "tent":

#                 n_adapt = tent_adapt(

#                     yolo.model,

#                     adap_batches,

#                     device=device,

#                     max_batches=len(adap_batches),

#                     steps_per_batch=args.tta_step,

#                     lr=args.tta_lr,

#                 )

#             else:

#                 n_adapt = pseudo_label_adapt(

#                     yolo.model,

#                     adap_batches,

#                     device=device,

#                     max_batches=len(adap_batches),

#                     steps_per_batch=args.tta_step,

#                     lr=args.tta_lr,

#                     conf_thr=args.tta_conf_thr,

#                 )

#             print(f"TTA mode={args.tta_mode}, adapted on {n_adapt} mini-batches.")

#     else:

#         print("TTA mode=none, evaluating baseline model.")

#         n_adapt = 0
 
#     # ------------------------------------------------------------------

#     # 2) Set up tracking + video writer + CSV

#     # ------------------------------------------------------------------

#     cap = cv2.VideoCapture(str(src_path))

#     if not cap.isOpened():

#         raise RuntimeError(f"Could not open video for fps/size: {src_path}")

#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

#     ret, first_frame = cap.read()

#     cap.release()

#     if not ret:

#         raise RuntimeError("Failed to read first frame for size.")

#     h, w = first_frame.shape[:2]

#     frame_size = (w, h)
 
#     out_video_path = Path(args.out_video)

#     writer = get_video_writer(out_video_path, frame_size, fps)
 
#     out_csv_path = Path(args.out_csv)

#     out_csv_path.parent.mkdir(parents=True, exist_ok=True)

#     csv_f = open(out_csv_path, "w", newline="")

#     csv_writer = csv.writer(csv_f)

#     csv_writer.writerow(

#         [

#             "frame",

#             "track_id",

#             "cls",

#             "conf",

#             "x1",

#             "y1",

#             "x2",

#             "y2",

#         ]

#     )
 
#     # ------------------------------------------------------------------

#     # 3) YOLO.track with ByteTrack, stream results frame by frame

#     # ------------------------------------------------------------------

#     frame_idx = 0

#     results = yolo.track(

#         source=str(src_path),

#         imgsz=args.imgsz,

#         conf=args.conf,

#         iou=args.iou,

#         device=str(device).replace("cuda:", "cuda"),

#         tracker=args.tracker,

#         stream=True,

#         verbose=False,

#     )
 
#     try:

#         for r in results:

#             # r: ultralytics.engine.results.Results

#             plotted = r.plot()  # BGR numpy array

#             writer.write(plotted)
 
#             boxes = r.boxes

#             if boxes is not None:

#                 xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]

#                 cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None

#                 conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None

#                 ids = boxes.id.cpu().numpy() if boxes.id is not None else None
 
#                 n = xyxy.shape[0]

#                 for i in range(n):

#                     tid = int(ids[i]) if ids is not None else -1

#                     c = int(cls[i]) if cls is not None else -1

#                     cf = float(conf[i]) if conf is not None else 0.0

#                     x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]

#                     csv_writer.writerow(

#                         [frame_idx, tid, c, cf, x1, y1, x2, y2]

#                     )
 
#             frame_idx += 1

#     finally:

#         writer.release()

#         csv_f.close()
 
#     print(f"Tracking complete. Video saved to {out_video_path.resolve()}")

#     print(f"Tracks CSV saved to {out_csv_path.resolve()}")

#     print(f"TTA mode={args.tta_mode}, n_adapt_batches={n_adapt}")
 
 
# if __name__ == "__main__":

#     main()

#-----------------speed--------------------

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from adapters import adabn_update, tent_adapt, pseudo_label_adapt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLOv8+ByteTrack on a video with optional TTA."
    )
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file (e.g. MOT17-02-FRCNN.mp4).",
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO weights (e.g. best.pt).",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Ultralytics (e.g. 'cuda', 'cpu', '0').",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config for YOLO.track (default ByteTrack).",
    )
    p.add_argument(
        "--tta-mode",
        choices=["none", "adabn", "tent", "pseudo"],
        default="adabn",
        help="TTA variant to apply before tracking.",
    )
    p.add_argument(
        "--tta-frames",
        type=int,
        default=64,
        help="Number of frames from the video to use for adaptation.",
    )
    p.add_argument(
        "--tta-step",
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
        "--out-video",
        type=str,
        default="outputs/tta/tta_track.mp4",
        help="Path to save annotated output video.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="outputs/tta/tta_tracks.csv",
        help="Path to save tracks CSV.",
    )
    return p.parse_args()


def sample_frames_for_tta(
    video_path: str, imgsz: int, n_frames: int, device: torch.device
) -> List[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = n_frames

    idxs = np.linspace(0, frame_count - 1, num=min(n_frames, frame_count), dtype=int)
    idx_set = set(int(i) for i in idxs.tolist())

    batches: List[torch.Tensor] = []
    cur = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idx_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            rgb = rgb.astype(np.float32) / 255.0
            rgb = np.transpose(rgb, (2, 0, 1))
            t = torch.from_numpy(rgb).unsqueeze(0).to(device, non_blocking=True)
            batches.append(t)
            if len(batches) >= n_frames:
                break
        cur += 1

    cap.release()
    return batches


def get_video_writer(
    out_path: Path, frame_size: Tuple[int, int], fps: float
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    return writer


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else "cuda")

    src_path = Path(args.source)
    if not src_path.exists():
        raise FileNotFoundError(f"Source video not found: {src_path}")

    yolo = YOLO(args.weights)
    yolo.to(device)

    if args.tta_mode != "none":
        adap_batches = sample_frames_for_tta(
            str(src_path), args.imgsz, args.tta_frames, device
        )
        if not adap_batches:
            print("No frames collected for TTA; skipping adaptation.")
            n_adapt = 0
        else:
            if args.tta_mode == "adabn":
                n_adapt = adabn_update(
                    yolo.model, adap_batches, device=device, max_batches=len(adap_batches)
                )
            elif args.tta_mode == "tent":
                n_adapt = tent_adapt(
                    yolo.model,
                    adap_batches,
                    device=device,
                    max_batches=len(adap_batches),
                    steps_per_batch=args.tta_step,
                    lr=args.tta_lr,
                )
            else:
                n_adapt = pseudo_label_adapt(
                    yolo.model,
                    adap_batches,
                    device=device,
                    max_batches=len(adap_batches),
                    steps_per_batch=args.tta_step,
                    lr=args.tta_lr,
                    conf_thr=args.tta_conf_thr,
                )
            print(f"TTA mode={args.tta_mode}, adapted on {n_adapt} mini-batches.")
    else:
        print("TTA mode=none, evaluating baseline model.")
        n_adapt = 0

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for fps/size: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame for size.")
    h, w = first_frame.shape[:2]
    frame_size = (w, h)

    out_video_path = Path(args.out_video)
    writer = get_video_writer(out_video_path, frame_size, fps)

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(
        ["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]
    )

    frame_idx = 0
    results = yolo.track(
        source=str(src_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=str(device).replace("cuda:", "cuda"),
        tracker=args.tracker,
        stream=True,
        verbose=False,
    )

    try:
        for r in results:
            plotted = r.plot()
            writer.write(plotted)

            boxes = r.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
                conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                n = xyxy.shape[0]
                for i in range(n):
                    tid = int(ids[i]) if ids is not None else -1
                    c = int(cls[i]) if cls is not None else -1
                    cf = float(conf[i]) if conf is not None else 0.0
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                    csv_writer.writerow(
                        [frame_idx, tid, c, cf, x1, y1, x2, y2]
                    )

            frame_idx += 1
    finally:
        writer.release()
        csv_f.close()

    print(f"Tracking complete. Video saved to {out_video_path.resolve()}")
    print(f"Tracks CSV saved to {out_csv_path.resolve()}")
    print(f"TTA mode={args.tta_mode}, n_adapt_batches={n_adapt}")


if __name__ == "__main__":
    main()
 