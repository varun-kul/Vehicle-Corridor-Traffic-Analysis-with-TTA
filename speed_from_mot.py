from __future__ import annotations
import argparse, csv, math, json
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import cv2
from ultralytics import YOLO
import torch

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO indices)


def apply_homography(xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    xy1 = np.concatenate([xy, np.ones((xy.shape[0],1))], axis=1)
    out = (H @ xy1.T).T
    out = out[:, :2] / np.clip(out[:, 2:], 1e-6, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='video path or camera index')
    ap.add_argument('--weights', default='outputs/detection/yolov8n_visdrone/weights/best.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='')
    ap.add_argument('--classes', type=int, nargs='*', default=VEHICLE_CLASSES)
    ap.add_argument('--fps', type=float, default=0.0, help='override FPS if video metadata is wrong')
    # Scaling options
    ap.add_argument('--meters_per_px', type=float, default=0.0, help='constant ground‑plane scale (m/px)')
    ap.add_argument('--homography_json', type=str, default='', help='JSON with {"H": [[...],[...],[...]]} (pixels→meters)')
    # Output
    ap.add_argument('--save_vid', action='store_true')
    ap.add_argument('--out_video', default='outputs/speed_annotated.mp4')
    ap.add_argument('--out_csv', default='outputs/speed_metrics.csv')
    ap.add_argument('--smooth', type=int, default=6, help='frames window for speed smoothing')
    args = ap.parse_args()

    model = YOLO(args.weights)

    # Homography or scale
    H = None
    if args.homography_json:
        with open(args.homography_json, 'r') as f:
            H = np.array(json.load(f)['H'], dtype=float)

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open source: {args.source}')
    fps = args.fps if args.fps > 0 else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Himg = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    vw = None
    if args.save_vid:
        Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(args.out_video, fourcc, fps, (W, Himg))

    # Per‑ID history for speed
    trails = defaultdict(lambda: deque(maxlen=max(2, args.smooth)))  # tid -> deque[(x,y)]
    speeds = {}  # tid -> km/h

    # CSV logger
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(args.out_csv, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['frame','track_id','cls','x1','y1','x2','y2','speed_kmh'])

    # Tracking stream (ByteTrack)
    stream = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=args.classes,
        tracker='bytetrack.yaml',
        persist=True,
        stream=True,
        verbose=True,
    )

    frame_idx = -1
    for res in stream:
        frame_idx += 1
        frame = res.orig_img.copy()

        if res.boxes is None or res.boxes.xyxy.numel() == 0:
            if args.save_vid and vw: vw.write(frame)
            cv2.imshow('Speed', frame) if not args.save_vid else None
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        ids   = res.boxes.id
        ids = ids.cpu().numpy().astype(int) if ids is not None else np.arange(len(xyxy))

        for box, cls, tid in zip(xyxy, clses, ids):
            x1,y1,x2,y2 = box.astype(int)
            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
            trails[tid].append((cx, cy))

            # speed calculation
            speed_kmh = None
            if len(trails[tid]) >= 2:
                p1 = np.array(trails[tid][-2]); p2 = np.array(trails[tid][-1])
                if H is not None:
                    m = apply_homography(np.vstack([p1, p2]).astype(float), H)
                    dist_m = float(np.linalg.norm(m[1]-m[0]))
                elif args.meters_per_px > 0:
                    dist_px = float(np.linalg.norm(p2-p1))
                    dist_m = dist_px * args.meters_per_px
                else:
                    dist_m = 0.0
                v_ms = dist_m * fps
                speed_kmh = v_ms * 3.6
                speeds[tid] = 0.7*speeds.get(tid, speed_kmh) + 0.3*speed_kmh  # EMA smoothing

            # draw
            color = (0, 200, 255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            lab = f"ID {tid} cls {cls}"
            if tid in speeds:
                lab += f" {speeds[tid]:.1f} km/h"
            cv2.putText(frame, lab, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.writerow([frame_idx, tid, cls, x1, y1, x2, y2, f"{speeds.get(tid, '')}"])

        # show / save
        if args.save_vid and vw:
            vw.write(frame)
        else:
            cv2.imshow('Speed', frame)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

    if vw: vw.release()
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()