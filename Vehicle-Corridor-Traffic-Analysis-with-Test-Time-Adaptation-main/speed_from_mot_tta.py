# from __future__ import annotations

# import argparse
# import csv
# from pathlib import Path
# from typing import Dict, Tuple, List

# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO

# from adapters import adabn_update, tent_adapt, pseudo_label_adapt


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description=(
#             "Run YOLOv8+ByteTrack with optional TTA and per-track speed overlay "
#             "using a global meters-per-pixel scale."
#         )
#     )
#     p.add_argument(
#         "--source",
#         type=str,
#         required=True,
#         help="Path to input video file.",
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
#         "--meters-per-px",
#         type=float,
#         default=None,
#         help=(
#             "Global m/px scale for converting image displacement to meters. "
#             "If not set, speeds are reported in px/s only."
#         ),
#     )
#     p.add_argument(
#         "--smooth-alpha",
#         type=float,
#         default=0.5,
#         help="EMA smoothing factor for speed (0=no smoothing, 1=no history).",
#     )
#     p.add_argument(
#         "--out-video",
#         type=str,
#         default="outputs/speed_tta/speed_tta.mp4",
#         help="Path to save annotated output video.",
#     )
#     p.add_argument(
#         "--out-csv",
#         type=str,
#         default="outputs/speed_tta/speed_tta.csv",
#         help="Path to save per-frame speeds CSV.",
#     )
#     return p.parse_args()


# def sample_frames_for_tta(
#     video_path: str, imgsz: int, n_frames: int, device: torch.device
# ) -> List[torch.Tensor]:
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
#             rgb = np.transpose(rgb, (2, 0, 1))
#             t = torch.from_numpy(rgb).unsqueeze(0).to(device, non_blocking=True)
#             batches.append(t)
#             if len(batches) >= n_frames:
#                 break
#         cur += 1

#     cap.release()
#     return batches


# def get_video_meta(video_path: str) -> Tuple[float, Tuple[int, int]]:
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {video_path}")
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps is None or fps <= 0:
#         fps = 25.0
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         raise RuntimeError("Could not read first frame for size.")
#     h, w = frame.shape[:2]
#     return float(fps), (w, h)


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

#     meters_per_px = args.meters_per_px
#     if meters_per_px is not None:
#         print(f"Using global scale: {meters_per_px:.6f} m/px.")
#     else:
#         print("No meters-per-px given; speeds will be in px/s only.")

#     yolo = YOLO(args.weights)
#     yolo.to(device)

#     # ---------------------- TTA adaptation ----------------------------
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
#         print("TTA mode=none, running baseline tracking+speed.")
#         n_adapt = 0

#     # ---------------------- Video meta + writers ----------------------
#     fps, frame_size = get_video_meta(str(src_path))
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
#             "speed_px_per_s",
#             "speed_kmh",
#         ]
#     )

#     # state per track_id: (last_frame, last_cx, last_cy, last_speed_kmh)
#     track_state: Dict[int, Tuple[int, float, float, float]] = {}

#     frame_idx = 0
#     try:
#         results = yolo.track(
#             source=str(src_path),
#             imgsz=args.imgsz,
#             conf=args.conf,
#             iou=args.iou,
#             device=str(device).replace("cuda:", "cuda"),
#             tracker=args.tracker,
#             stream=True,
#             verbose=False,
#         )

#         for r in results:
#             plotted = r.plot()
#             boxes = r.boxes
#             if boxes is not None:
#                 xyxy = boxes.xyxy.cpu().numpy()
#                 cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
#                 conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
#                 ids = boxes.id.cpu().numpy() if boxes.id is not None else None

#                 n = xyxy.shape[0]
#                 for i in range(n):
#                     tid = int(ids[i]) if ids is not None else -1
#                     c = int(cls[i]) if cls is not None else -1
#                     cf = float(conf[i]) if conf is not None else 0.0
#                     x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
#                     cx = 0.5 * (x1 + x2)
#                     cy = 0.5 * (y1 + y2)

#                     speed_px_per_s = 0.0
#                     speed_kmh = 0.0

#                     if tid in track_state:
#                         last_f, last_cx, last_cy, last_speed = track_state[tid]
#                         df = max(frame_idx - last_f, 1)
#                         dx = cx - last_cx
#                         dy = cy - last_cy
#                         dist_px = float(np.hypot(dx, dy))
#                         speed_px_per_s = dist_px * fps / df

#                         if meters_per_px is not None:
#                             dist_m = dist_px * meters_per_px
#                             speed_mps_raw = dist_m * fps / df
#                             speed_kmh_raw = speed_mps_raw * 3.6
#                             if last_speed is None:
#                                 speed_kmh = speed_kmh_raw
#                             else:
#                                 alpha = float(args.smooth_alpha)
#                                 speed_kmh = (
#                                     alpha * speed_kmh_raw
#                                     + (1.0 - alpha) * last_speed
#                                 )
#                         else:
#                             speed_kmh = 0.0

#                     else:
#                         last_speed = None

#                     track_state[tid] = (frame_idx, cx, cy, speed_kmh)

#                     if meters_per_px is not None:
#                         label = f"{speed_kmh:.1f} km/h"
#                         # to use mph instead:
#                         # speed_mph = speed_kmh * 0.621371
#                         # label = f"{speed_mph:.1f} mph"
#                     else:
#                         label = f"{speed_px_per_s:.1f} px/s"

#                     text_pos = (int(x1), int(max(y1 - 5, 0)))
#                     cv2.putText(
#                         plotted,
#                         label,
#                         text_pos,
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 255, 255),
#                         1,
#                         cv2.LINE_AA,
#                     )

#                     csv_writer.writerow(
#                         [
#                             frame_idx,
#                             tid,
#                             c,
#                             cf,
#                             x1,
#                             y1,
#                             x2,
#                             y2,
#                             speed_px_per_s,
#                             speed_kmh,
#                         ]
#                     )

#             writer.write(plotted)
#             frame_idx += 1

#     finally:
#         writer.release()
#         csv_f.close()

#     print(f"Done. Video with speeds: {out_video_path.resolve()}")
#     print(f"Per-frame speed CSV: {out_csv_path.resolve()}")
#     print(f"TTA mode={args.tta_mode}, n_adapt_batches={n_adapt}")


# if __name__ == "__main__":
#     main()


#-------------------------------

# speed_from_mot_tta.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from adapters import adabn_update, tent_adapt, pseudo_label_adapt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run YOLOv8+ByteTrack with optional TTA and per-track speed overlay "
            "using a global meters-per-pixel scale."
        )
    )
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file.",
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
        "--meters-per-px",
        type=float,
        default=None,
        help=(
            "Global m/px scale for converting image displacement to meters. "
            "If not set, speeds are reported in px/s only."
        ),
    )
    p.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="EMA smoothing factor for speed (0=no smoothing, 1=no history).",
    )
    p.add_argument(
        "--out-video",
        type=str,
        default="outputs/speed_tta/speed_tta.mp4",
        help="Path to save annotated output video.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="outputs/speed_tta/speed_tta.csv",
        help="Path to save per-frame speeds CSV.",
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


def get_video_meta(video_path: str) -> Tuple[float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame for size.")
    h, w = frame.shape[:2]
    return float(fps), (w, h)


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

    meters_per_px = args.meters_per_px
    if meters_per_px is not None:
        print(f"Using global scale: {meters_per_px:.6f} m/px.")
    else:
        print("No meters-per-px given; speeds will be in px/s only.")

    # Load detector
    yolo = YOLO(args.weights)
    yolo.to(device)

    # ---------------------- TTA adaptation ----------------------------
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
        print("TTA mode=none, running baseline tracking+speed.")
        n_adapt = 0

    # ---------------------- Video meta + writers ----------------------
    fps, frame_size = get_video_meta(str(src_path))
    out_video_path = Path(args.out_video)
    writer = get_video_writer(out_video_path, frame_size, fps)

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(
        [
            "frame",
            "track_id",
            "cls",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "speed_px_per_s",
            "speed_kmh",
        ]
    )

    # state per track_id: (last_frame, last_cx, last_cy, last_speed_kmh)
    track_state: Dict[int, Tuple[int, float, float, float]] = {}

    frame_idx = 0
    try:
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

        for r in results:
            # Start from original frame; we will draw boxes, id, class, speed ourselves
            frame = r.orig_img.copy()  # BGR
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
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)

                    speed_px_per_s = 0.0
                    speed_kmh = 0.0

                    if tid in track_state:
                        last_f, last_cx, last_cy, last_speed = track_state[tid]
                        df = max(frame_idx - last_f, 1)
                        dx = cx - last_cx
                        dy = cy - last_cy
                        dist_px = float(np.hypot(dx, dy))
                        speed_px_per_s = dist_px * fps / df

                        if meters_per_px is not None:
                            dist_m = dist_px * meters_per_px
                            speed_mps_raw = dist_m * fps / df
                            speed_kmh_raw = speed_mps_raw * 3.6
                            if last_speed is None:
                                speed_kmh = speed_kmh_raw
                            else:
                                alpha = float(args.smooth_alpha)
                                speed_kmh = (
                                    alpha * speed_kmh_raw
                                    + (1.0 - alpha) * last_speed
                                )
                        else:
                            speed_kmh = 0.0
                    else:
                        last_speed = None

                    # update state
                    track_state[tid] = (frame_idx, cx, cy, speed_kmh)

                    # -------- label: id + class + speed on the same line --------
                    if hasattr(yolo, "names") and c in getattr(yolo, "names", {}):
                        class_name = yolo.names[c]
                    else:
                        class_name = str(c)

                    tid_str = "NA" if tid < 0 else str(tid)

                    if meters_per_px is not None:
                        speed_str = f"{speed_kmh:.1f} km/h"
                        # for mph instead:
                        # speed_mph = speed_kmh * 0.621371
                        # speed_str = f"{speed_mph:.1f} mph"
                    else:
                        speed_str = f"{speed_px_per_s:.1f} px/s"

                    label_text = f"id={tid_str} {class_name} {speed_str}"

                    # -------------- draw box --------------
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    color = (0, 255, 0)
                    cv2.rectangle(frame, pt1, pt2, color, 2)

                    # -------------- draw text background + text --------------
                    (tw, th), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    # top-left corner for text background
                    text_bg_tl = (pt1[0], max(pt1[1] - th - baseline - 4, 0))
                    text_bg_br = (pt1[0] + tw + 4, pt1[1])

                    cv2.rectangle(frame, text_bg_tl, text_bg_br, color, thickness=-1)
                    text_org = (text_bg_tl[0] + 2, pt1[1] - baseline - 2)
                    cv2.putText(
                        frame,
                        label_text,
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    csv_writer.writerow(
                        [
                            frame_idx,
                            tid,
                            c,
                            cf,
                            x1,
                            y1,
                            x2,
                            y2,
                            speed_px_per_s,
                            speed_kmh,
                        ]
                    )

            writer.write(frame)
            frame_idx += 1

    finally:
        writer.release()
        csv_f.close()

    print(f"Done. Video with speeds: {out_video_path.resolve()}")
    print(f"Per-frame speed CSV: {out_csv_path.resolve()}")
    print(f"TTA mode={args.tta_mode}, n_adapt_batches={n_adapt}")


if __name__ == "__main__":
    main()
