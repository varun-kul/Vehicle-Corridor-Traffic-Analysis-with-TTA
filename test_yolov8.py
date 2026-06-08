from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO


def results_to_rows(res):
    rows = []
    p = Path(res.path)
    if res.boxes is None or res.boxes.xyxy.numel() == 0:
        return rows
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls  = res.boxes.cls.cpu().numpy()
    for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
        rows.append({
            'image_path': str(p),
            'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
            'conf': float(c), 'cls': int(k)
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='outputs/detection/yolov8n_visdrone/weights/best.pt')
    ap.add_argument('--data', default='data/visdrone.yaml', help='dataset yaml; must include test path')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='')
    ap.add_argument('--out_dir', default='outputs/preds/test')
    ap.add_argument('--eval_if_labels', action='store_true', help='if test has labels, also compute metrics')
    args = ap.parse_args()

    model = YOLO(args.weights)

    # Predict across test set; also save YOLO TXT (for submission/inspection)
    results = model.predict(
        data=args.data,
        split='test',
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_txt=True,
        save_conf=True,
        project=args.out_dir,
        name='txt',
        verbose=True,
    )

    # Consolidate predictions to CSV
    all_rows = []
    for r in results:
        all_rows.extend(results_to_rows(r))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'predictions.csv'
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved CSV predictions → {csv_path}")

    # Optional evaluation if labels exist in the YAML's test set
    if args.eval_if_labels:
        metrics = model.val(data=args.data, split='test', imgsz=args.imgsz, device=args.device, conf=0.001, iou=0.6)
        payload = {
            'map50_95': metrics.box.map,
            'map50': metrics.box.map50,
            'maps_per_class': list(metrics.box.maps),
        }
        with open(out_dir / 'test_metrics.json', 'w') as f:
            json.dump(payload, f, indent=2)
        print('Test metrics:', payload)


if __name__ == '__main__':
    main()