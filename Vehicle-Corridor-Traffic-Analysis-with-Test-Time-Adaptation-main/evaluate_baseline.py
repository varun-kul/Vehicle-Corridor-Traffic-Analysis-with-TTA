# """
# Quick baseline evaluation on VisDrone
# """
# import sys
# sys.path.append('src')

# from pathlib import Path
# import pandas as pd
# import json
# from ultralytics import YOLO
# from tqdm import tqdm
# import numpy as np

# # Load model
# model = YOLO('outputs/detection/yolov8n_visdrone3/weights/best.pt')

# # Evaluate on validation set
# print("Evaluating on validation set...")
# metrics = model.val(
#     data='data/processed/yolo/visdrone.yaml',
#     split='val'
# )

# results = {
#     'overall': {
#         'mAP50': float(metrics.box.map50),
#         'mAP50-95': float(metrics.box.map),
#         'precision': float(metrics.box.mp),
#         'recall': float(metrics.box.mr)
#     }
# }

# print(f"\n{'='*70}")
# print("BASELINE EVALUATION RESULTS")
# print(f"{'='*70}")
# print(f"mAP@0.5: {results['overall']['mAP50']:.4f}")
# print(f"mAP@0.5:0.95: {results['overall']['mAP50-95']:.4f}")
# print(f"Precision: {results['overall']['precision']:.4f}")
# print(f"Recall: {results['overall']['recall']:.4f}")

# # Evaluate by condition
# print(f"\n{'='*70}")
# print("EVALUATING BY CONDITION")
# print(f"{'='*70}")

# conditions = ['day_clear', 'night_clear', 'day_cloudy', 'day_foggy']
# condition_results = {}

# for condition in conditions:
#     anno_file = Path(f"data/processed/train/annotations_{condition}.csv")
    
#     if not anno_file.exists():
#         print(f"\nSkipping {condition} (no data)")
#         continue
    
#     print(f"\n{condition}:")
#     df = pd.read_csv(anno_file)
    
#     # Sample evaluation on subset
#     images = df['image_id'].unique()[:50]
    
#     tp, fp, fn = 0, 0, 0
    
#     for img_id in tqdm(images[:50], desc=f"  {condition}"):
#         img_annos = df[df['image_id'] == img_id]
#         img_path = Path("data/raw/visdrone") / img_annos.iloc[0]['image_path']
        
#         if not img_path.exists():
#             continue
        
#         # Run detection
#         preds = model(img_path, verbose=False)[0]
        
#         gt_boxes = img_annos[['x1', 'y1', 'x2', 'y2']].values
        
#         if len(preds.boxes) > 0:
#             pred_boxes = preds.boxes.xyxy.cpu().numpy()
            
#             # Simple matching
#             matched = set()
#             for pred_box in pred_boxes:
#                 best_iou = 0
#                 best_idx = -1
#                 for i, gt_box in enumerate(gt_boxes):
#                     if i in matched:
#                         continue
                    
#                     # IOU
#                     x1 = max(pred_box[0], gt_box[0])
#                     y1 = max(pred_box[1], gt_box[1])
#                     x2 = min(pred_box[2], gt_box[2])
#                     y2 = min(pred_box[3], gt_box[3])
                    
#                     inter = max(0, x2-x1) * max(0, y2-y1)
#                     area1 = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
#                     area2 = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
#                     union = area1 + area2 - inter
#                     iou = inter / union if union > 0 else 0
                    
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_idx = i
                
#                 if best_iou > 0.5:
#                     tp += 1
#                     matched.add(best_idx)
#                 else:
#                     fp += 1
            
#             fn += len(gt_boxes) - len(matched)
#         else:
#             fn += len(gt_boxes)
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
#     condition_results[condition] = {
#         'precision': precision,
#         'recall': recall,
#         'tp': tp,
#         'fp': fp,
#         'fn': fn
#     }
    
#     print(f"  Precision: {precision:.4f}")
#     print(f"  Recall: {recall:.4f}")

# results['by_condition'] = condition_results

# # Save results
# output_dir = Path('outputs/metrics')
# output_dir.mkdir(parents=True, exist_ok=True)

# with open(output_dir / 'baseline_results.json', 'w') as f:
#     json.dump(results, f, indent=2)

# print(f"\n{'='*70}")
# print("RESULTS SAVED")
# print(f"{'='*70}")
# print(f"Saved to: outputs/metrics/baseline_results.json")

# # Calculate domain shift
# if 'day_clear' in condition_results and 'night_clear' in condition_results:
#     day_prec = condition_results['day_clear']['precision']
#     night_prec = condition_results['night_clear']['precision']
#     degradation = (day_prec - night_prec) / day_prec * 100
    
#     print(f"\n🔍 DOMAIN SHIFT ANALYSIS:")
#     print(f"  Day→Night degradation: {degradation:.1f}%")
#     print(f"  This is what Phase 2 TTA will address!")

# print(f"\n{'='*70}")
# print("PHASE 1 BASELINE COMPLETE! ✅")
# print(f"{'='*70}")
# print("\nNext: Implement Phase 2 (Test-Time Adaptation)")

#----------------------------------------------------

"""
Evaluate tracking performance on MOT17 using MOT metrics
"""
import sys
sys.path.append('src')

from pathlib import Path
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracking.deepsort_tracker import DeepSORT
import motmetrics as mm
import cv2
from tqdm import tqdm
import json

def evaluate_mot_sequence(model, tracker, sequence_path):
    """Evaluate on single MOT17 sequence"""
    
    # Load ground truth
    gt_file = sequence_path / 'gt' / 'gt.txt'
    
    if not gt_file.exists():
        return None
    
    # Read ground truth
    gt_df = pd.read_csv(
        gt_file,
        header=None,
        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    )
    
    # Filter only valid detections (class 1 = pedestrian, but we'll track all)
    gt_df = gt_df[gt_df['conf'] == 1]  # Only consider valid annotations
    
    # Create accumulator for metrics
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Get image directory
    img_dir = sequence_path / 'img1'
    frames = sorted([int(f.stem) for f in img_dir.glob('*.jpg')])
    
    print(f"  Processing {len(frames)} frames...")
    
    # Process each frame
    for frame_num in tqdm(frames[:500], desc=f"  {sequence_path.name}"):  # Limit to 500 frames
        
        # Load image
        img_path = img_dir / f"{frame_num:06d}.jpg"
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            continue
        
        # Get ground truth for this frame
        gt_frame = gt_df[gt_df['frame'] == frame_num]
        gt_ids = gt_frame['id'].values
        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        
        # Convert to [x1,y1,x2,y2]
        gt_boxes_xyxy = np.zeros_like(gt_boxes)
        gt_boxes_xyxy[:, 0] = gt_boxes[:, 0]
        gt_boxes_xyxy[:, 1] = gt_boxes[:, 1]
        gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
        gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
        
        # Run detection
        results = model(frame, verbose=False)[0]
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            detections = np.column_stack([boxes, scores])
        else:
            detections = np.empty((0, 5))
        
        # Run tracking
        tracks = tracker.update(detections, frame)
        
        # Get predicted IDs and boxes
        if len(tracks) > 0:
            pred_ids = tracks[:, 4].astype(int)
            pred_boxes = tracks[:, :4]
        else:
            pred_ids = np.array([])
            pred_boxes = np.empty((0, 4))
        
        # Compute distances (using IOU distance)
        if len(gt_boxes_xyxy) > 0 and len(pred_boxes) > 0:
            distances = mm.distances.iou_matrix(gt_boxes_xyxy, pred_boxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_boxes_xyxy), len(pred_boxes)))
        
        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)
    
    return acc

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("="*70)
    print("MOT17 TRACKING EVALUATION")
    print("="*70)
    
    # Load model
    model_path = 'outputs/detection/yolov8n_visdrone3/weights/best.pt'
    
    if not Path(model_path).exists():
        print("❌ Model not found!")
        exit(1)
    
    model = YOLO(model_path)
    print("✅ Loaded detector")
    
    # Get MOT17 sequences
    mot_base = Path('data/raw/mot17/MOT17/train')
    
    if not mot_base.exists():
        print("❌ MOT17 dataset not found!")
        exit(1)
    
    sequences = sorted([s for s in mot_base.glob('MOT17-*-FRCNN')])[:3]  # Evaluate on 3 sequences
    
    print(f"\nEvaluating on {len(sequences)} sequences...")
    
    # Evaluate each sequence
    all_accs = []
    
    for seq_path in sequences:
        print(f"\n--- {seq_path.name} ---")
        
        # Initialize fresh tracker for each sequence
        tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
        
        acc = evaluate_mot_sequence(model, tracker, seq_path)
        
        if acc is not None:
            all_accs.append(acc)
    
    # Compute overall metrics
    if len(all_accs) > 0:
        mh = mm.metrics.create()
        
        summary = mh.compute_many(
            all_accs,
            metrics=['num_frames', 'mota', 'motp', 'idf1', 'num_switches', 
                    'num_false_positives', 'num_misses', 'precision', 'recall'],
            names=[s.name for s in sequences]
        )
        
        print("\n" + "="*70)
        print("TRACKING METRICS")
        print("="*70)
        print("\n" + str(summary))
        
        # Save results
        results = {
            'sequences_evaluated': [s.name for s in sequences],
            'metrics': summary.to_dict()
        }
        
        output_dir = Path('outputs/metrics')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        results_serializable = {
            'sequences_evaluated': results['sequences_evaluated'],
            'metrics': {}
        }
        
        for col in summary.columns:
            results_serializable['metrics'][col] = {}
            for idx in summary.index:
                val = summary.loc[idx, col]
                if pd.notna(val):
                    results_serializable['metrics'][col][idx] = float(val)
        
        with open(output_dir / 'mot17_evaluation.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✅ Results saved to: outputs/metrics/mot17_evaluation.json")
        
        # Print key metrics
        print("\n" + "="*70)
        print("KEY METRICS (Average)")
        print("="*70)
        if 'mota' in summary.columns:
            print(f"MOTA: {summary['mota'].mean():.4f}")
        if 'idf1' in summary.columns:
            print(f"IDF1: {summary['idf1'].mean():.4f}")
        if 'num_switches' in summary.columns:
            print(f"ID Switches: {summary['num_switches'].sum():.0f}")
        if 'precision' in summary.columns:
            print(f"Precision: {summary['precision'].mean():.4f}")
        if 'recall' in summary.columns:
            print(f"Recall: {summary['recall'].mean():.4f}")
        
        print("="*70)
        print("\n✅ MOT17 evaluation complete!")
    else:
        print("\n❌ No sequences could be evaluated")