from __future__ import annotations
import argparse, json
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='outputs/detection/yolov8n_visdrone/weights/best.pt')
    ap.add_argument('--data', default='data/visdrone.yaml', help='dataset yaml with train/val/test paths')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='')  # 'cpu' or 'cuda:0' or '' for auto
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--out', default='outputs/metrics/val_metrics.json')
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        verbose=True,
    )

    # Collect high‑level box metrics
    payload = {
        'weights': str(args.weights),
        'data': str(args.data),
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'conf': args.conf,
        'iou': args.iou,
        'metrics': {
            'map50_95': metrics.box.map,      # mAP@[.5:.95]
            'map50': metrics.box.map50,
            'maps_per_class': list(metrics.box.maps),
            'speed_ms': dict(metrics.speed),  # {'preprocess':..,'inference':..,'postprocess':..}
        }
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload['metrics'], indent=2))
    print(f"Saved metrics → {outp}")


if __name__ == '__main__':
    main()