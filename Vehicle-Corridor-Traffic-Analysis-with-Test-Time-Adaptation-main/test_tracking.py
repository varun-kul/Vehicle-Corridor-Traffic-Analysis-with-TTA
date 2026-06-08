# """
# Test tracking with your trained model
# """
# import cv2
# import numpy as np
# from pathlib import Path
# import sys
# sys.path.append('src')

# from ultralytics import YOLO
# from tracking.deepsort_tracker import DeepSORT

# # Load your trained detector
# model = YOLO('outputs/detection/yolov8n_visdrone3/weights/best.pt')

# # Initialize tracker
# tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)

# # Get MOT17 sequence (or use VisDrone images as sequence)
# video_dir = Path('data/raw/mot17/MOT17/train/MOT17-02-FRCNN/img1')

# if not video_dir.exists():
#     print("MOT17 not found, using VisDrone images...")
#     # Use VisDrone validation images as a sequence
#     images = sorted(list(Path('data/raw/visdrone/VisDrone2019-DET-val/images').glob('*.jpg')))[:100]
# else:
#     images = sorted(list(video_dir.glob('*.jpg')))[:100]

# # Process sequence
# frame_num = 0
# total_tracks = set()

# for img_path in images:
#     frame = cv2.imread(str(img_path))
#     frame_num += 1
    
#     # Detection
#     results = model(frame, verbose=False)[0]
    
#     if len(results.boxes) > 0:
#         boxes = results.boxes.xyxy.cpu().numpy()
#         scores = results.boxes.conf.cpu().numpy()
#         detections = np.column_stack([boxes, scores])
#     else:
#         detections = np.empty((0, 5))
    
#     # Tracking
#     tracks = tracker.update(detections, frame)
    
#     # Visualize
#     for track in tracks:
#         x1, y1, x2, y2, track_id = track
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         track_id = int(track_id)
        
#         total_tracks.add(track_id)
        
#         # Draw box and ID
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Show frame
#     cv2.imshow('Tracking', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# print(f"\nTracking Summary:")
# print(f"  Frames processed: {frame_num}")
# print(f"  Unique tracks: {len(total_tracks)}")
# print(f"  ✓ Tracking works!")

#----------------------------------------

# """
# Test tracking with your trained model
# """
# import cv2
# import numpy as np
# from pathlib import Path
# import sys

# # Fix import path
# sys.path.append('src')
# sys.path.append('.')

# try:
#     from src.tracking.deepsort_tracker import DeepSORT
# except:
#     from tracking.deepsort_tracker import DeepSORT

# from ultralytics import YOLO

# # Load your trained detector
# model = YOLO('outputs/detection/yolov8n_visdrone3/weights/best.pt')

# # Initialize tracker
# tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)

# # Get images for testing
# print("Looking for test images...")

# # Try MOT17 first
# video_dir = Path('data/raw/mot17/MOT17/train/MOT17-02-FRCNN/img1')

# if video_dir.exists():
#     print("Using MOT17 sequence...")
#     images = sorted(list(video_dir.glob('*.jpg')))[:100]
# else:
#     print("MOT17 not found, using VisDrone images...")
#     # Use VisDrone validation images as a sequence
#     val_dir = Path('data/raw/visdrone/VisDrone2019-DET-val/images')
#     if val_dir.exists():
#         images = sorted(list(val_dir.glob('*.jpg')))[:100]
#     else:
#         print("Error: No images found!")
#         print("Please check:")
#         print("  - data/raw/visdrone/VisDrone2019-DET-val/images/")
#         print("  - data/raw/mot17/MOT17/train/")
#         exit(1)

# print(f"Found {len(images)} images")

# # Process sequence
# frame_num = 0
# total_tracks = set()

# print("\nProcessing frames (press 'q' to quit)...")

# for img_path in images:
#     frame = cv2.imread(str(img_path))
    
#     if frame is None:
#         continue
    
#     frame_num += 1
    
#     # Detection
#     results = model(frame, verbose=False)[0]
    
#     if len(results.boxes) > 0:
#         boxes = results.boxes.xyxy.cpu().numpy()
#         scores = results.boxes.conf.cpu().numpy()
#         detections = np.column_stack([boxes, scores])
#     else:
#         detections = np.empty((0, 5))
    
#     # Tracking
#     tracks = tracker.update(detections, frame)
    
#     # Visualize
#     for track in tracks:
#         x1, y1, x2, y2, track_id = track
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#         track_id = int(track_id)
        
#         total_tracks.add(track_id)
        
#         # Draw box and ID
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Add frame info
#     cv2.putText(frame, f"Frame: {frame_num} | Tracks: {len(tracks)}", 
#                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
#     # Show frame
#     cv2.imshow('Tracking Test', frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# print(f"\n{'='*60}")
# print("TRACKING TEST SUMMARY")
# print(f"{'='*60}")
# print(f"  Frames processed: {frame_num}")
# print(f"  Unique tracks: {len(total_tracks)}")
# print(f"  ✅ Tracking works!")
# print(f"{'='*60}")

#----------------------------------------------------------------------------------

"""
Test tracking with trained detector on MOT17
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')

from ultralytics import YOLO
from tracking.deepsort_tracker import DeepSORT
import json

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("="*70)
    print("TESTING TRACKING ON MOT17")
    print("="*70)
    
    # Load trained detector
    model_path = 'outputs/detection/yolov8n_visdrone3/weights/best.pt'
    
    if not Path(model_path).exists():
        print("❌ Model not found! Train first.")
        exit(1)
    
    model = YOLO(model_path)
    print(f"✅ Loaded detector: {model_path}")
    
    # Initialize tracker
    tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
    print("✅ Initialized DeepSORT tracker")
    
    # Get MOT17 sequence
    mot_base = Path('data/raw/mot17/MOT17/train')
    
    if not mot_base.exists():
        print(f"❌ MOT17 not found at: {mot_base}")
        exit(1)
    
    # Use first sequence
    sequences = sorted([s for s in mot_base.glob('MOT17-*-FRCNN')])
    
    if len(sequences) == 0:
        print("❌ No MOT17 sequences found!")
        exit(1)
    
    sequence = sequences[0]  # Use first sequence
    img_dir = sequence / 'img1'
    
    print(f"\nUsing sequence: {sequence.name}")
    
    # Get images
    images = sorted(list(img_dir.glob('*.jpg')))[:200]  # Process 200 frames
    print(f"Processing {len(images)} frames...")
    
    # Create output video
    output_path = f'outputs/tracking_test_{sequence.name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    # Tracking statistics
    frame_num = 0
    total_tracks = set()
    track_lengths = {}
    id_switches = 0
    prev_tracks = {}
    
    print("\nProcessing...")
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            continue
        
        frame_num += 1
        
        # Initialize video writer
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
        
        # Detection
        results = model(frame, verbose=False)[0]
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            detections = np.column_stack([boxes, scores])
        else:
            detections = np.empty((0, 5))
        
        # Tracking
        tracks = tracker.update(detections, frame)
        
        # Visualize and collect statistics
        current_tracks = {}
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            
            total_tracks.add(track_id)
            current_tracks[track_id] = (x1, y1, x2, y2)
            
            # Track length
            if track_id not in track_lengths:
                track_lengths[track_id] = 0
            track_lengths[track_id] += 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            label = f"ID:{track_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-label_size[1]-4), 
                         (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Detect ID switches (simplified)
        for tid in current_tracks:
            if tid in prev_tracks:
                # Check if position changed significantly (potential switch)
                curr_pos = current_tracks[tid]
                prev_pos = prev_tracks[tid]
                
                dist = np.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)
                if dist > 200:  # Large jump might indicate ID switch
                    id_switches += 1
        
        prev_tracks = current_tracks
        
        # Add frame info overlay
        info_y = 30
        cv2.putText(frame, f"Frame: {frame_num}/{len(images)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", (10, info_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total IDs: {len(total_tracks)}", (10, info_y+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % 50 == 0:
            print(f"  Processed {frame_num}/{len(images)} frames...")
    
    if out:
        out.release()
    
    # Calculate statistics
    avg_track_length = np.mean(list(track_lengths.values())) if track_lengths else 0
    
    # Save tracking results
    tracking_results = {
        'sequence': sequence.name,
        'total_frames': frame_num,
        'total_tracks': len(total_tracks),
        'avg_track_length': float(avg_track_length),
        'id_switches_detected': id_switches,
        'track_lengths': {str(k): v for k, v in track_lengths.items()}
    }
    
    output_dir = Path('outputs/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'tracking_results.json', 'w') as f:
        json.dump(tracking_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("TRACKING TEST SUMMARY")
    print("="*70)
    print(f"Sequence: {sequence.name}")
    print(f"Frames processed: {frame_num}")
    print(f"Total unique tracks: {len(total_tracks)}")
    print(f"Average track length: {avg_track_length:.1f} frames")
    print(f"ID switches detected: {id_switches}")
    print(f"\nOutput video: {output_path}")
    print(f"Results saved: outputs/metrics/tracking_results.json")
    print("="*70)
    print("\n✅ Tracking test complete!")