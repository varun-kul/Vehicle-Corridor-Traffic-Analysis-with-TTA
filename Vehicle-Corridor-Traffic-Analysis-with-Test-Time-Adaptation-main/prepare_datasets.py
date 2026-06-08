#!/usr/bin/env python3
"""
scripts/prepare_datasets_v2.py
Parse and prepare VisDrone and MOT17 datasets
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import configparser

class VisDroneParser:
    """
    Parse VisDrone dataset annotations
    
    VisDrone annotation format (space-separated):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    Object categories:
    0: ignored regions
    1: pedestrian
    2: people
    3: bicycle
    4: car
    5: van
    6: truck
    7: tricycle
    8: awning-tricycle
    9: bus
    10: motor
    11: others
    """
    
    def __init__(self, data_root="data/raw/visdrone"):
        self.data_root = Path(data_root)
        
        # Map VisDrone classes to our vehicle classes
        self.class_map = {
            4: 0,   # car -> car
            5: 1,   # van -> van
            9: 2,   # bus -> bus
            6: 3,   # truck -> truck
            7: 4,   # tricycle -> others
            8: 4,   # awning-tricycle -> others
            10: 4,  # motor -> others
            11: 4   # others -> others
        }
        
        self.class_names = ['car', 'van', 'bus', 'truck', 'others']
        
        # Simulate weather/illumination from image analysis
        # In VisDrone, we'll infer conditions from image statistics
    
    def parse_annotation_file(self, anno_path, img_path):
        """Parse single VisDrone annotation file"""
        annotations = []
        
        if not anno_path.exists():
            return annotations
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return annotations
        
        h, w = img.shape[:2]
        
        # Infer conditions from image (simple heuristics)
        illumination = self._infer_illumination(img)
        weather = self._infer_weather(img)
        
        # Read annotations
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            score = int(parts[4])
            category = int(parts[5])
            truncation = int(parts[6])
            occlusion = int(parts[7])
            
            # Filter: only keep vehicle classes
            if category not in self.class_map:
                continue
            
            # Filter: skip heavily occluded or truncated
            if occlusion > 2 or truncation > 1:
                continue
            
            # Skip invalid boxes
            if bbox_width <= 0 or bbox_height <= 0:
                continue
            
            x1 = bbox_left
            y1 = bbox_top
            x2 = bbox_left + bbox_width
            y2 = bbox_top + bbox_height
            
            annotations.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': bbox_width,
                'height': bbox_height,
                'class_id': self.class_map[category],
                'class': self.class_names[self.class_map[category]],
                'truncation': truncation,
                'occlusion': occlusion,
                'illumination': illumination,
                'weather': weather
            })
        
        return annotations
    
    def _infer_illumination(self, img):
        """Infer day/night from image brightness"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Simple threshold: < 80 is night, >= 80 is day
        return 'night' if mean_brightness < 80 else 'day'
    
    def _infer_weather(self, img):
        """Infer weather from image characteristics"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Check saturation (fog/rain has low saturation)
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)
        
        # Check contrast (fog reduces contrast)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        if mean_sat < 40 and contrast < 40:
            return 'foggy'
        elif mean_sat < 60:
            return 'cloudy'
        else:
            return 'clear'
    
    def process_split(self, split_name='train'):
        """Process one split (train/val/test-dev)"""
        print(f"\nProcessing VisDrone {split_name} split...")
        
        split_dir = self.data_root / f"VisDrone2019-DET-{split_name}"
        
        if not split_dir.exists():
            print(f"Error: Directory not found: {split_dir}")
            return None
        
        images_dir = split_dir / "images"
        annos_dir = split_dir / "annotations"
        
        if not images_dir.exists() or not annos_dir.exists():
            print(f"Error: Missing images or annotations directory")
            return None
        
        # Get all images
        image_files = sorted(list(images_dir.glob("*.jpg")))
        print(f"Found {len(image_files)} images")
        
        all_annotations = []
        
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            # Corresponding annotation file
            anno_path = annos_dir / f"{img_path.stem}.txt"
            
            # Parse annotations
            annotations = self.parse_annotation_file(anno_path, img_path)
            
            # Add image info to each annotation
            for anno in annotations:
                anno['image_id'] = img_path.stem
                anno['image_path'] = str(img_path.relative_to(self.data_root))
                all_annotations.append(anno)
        
        if len(all_annotations) == 0:
            print(f"Warning: No annotations found for {split_name}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_annotations)
        
        # Save processed annotations
        output_dir = Path(f"data/processed/{split_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_dir / "annotations.csv", index=False)
        
        print(f"Saved {len(df)} annotations")
        print(f"  Images: {df['image_id'].nunique()}")
        print(f"  Vehicles: {len(df)}")
        print(f"  Classes: {df['class'].value_counts().to_dict()}")
        
        # Create condition-based splits
        self.create_condition_splits(df, output_dir)
        
        return df
    
    def create_condition_splits(self, df, output_dir):
        """Split by inferred conditions"""
        print("Creating condition-based splits...")
        
        conditions = {
            'day_clear': df[(df['illumination'] == 'day') & (df['weather'] == 'clear')],
            'day_cloudy': df[(df['illumination'] == 'day') & (df['weather'] == 'cloudy')],
            'day_foggy': df[(df['illumination'] == 'day') & (df['weather'] == 'foggy')],
            'night_clear': df[(df['illumination'] == 'night') & (df['weather'] == 'clear')],
            'night_cloudy': df[(df['illumination'] == 'night') & (df['weather'] == 'cloudy')],
            'night_foggy': df[(df['illumination'] == 'night') & (df['weather'] == 'foggy')]
        }
        
        stats = {}
        for cond_name, cond_df in conditions.items():
            if len(cond_df) > 0:
                cond_df.to_csv(output_dir / f"annotations_{cond_name}.csv", index=False)
                stats[cond_name] = {
                    'num_images': cond_df['image_id'].nunique(),
                    'num_boxes': len(cond_df)
                }
                print(f"  {cond_name}: {len(cond_df)} boxes, {cond_df['image_id'].nunique()} images")
        
        with open(output_dir / "condition_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def run(self):
        """Process all splits"""
        print("="*70)
        print("VisDrone Dataset Preparation")
        print("="*70)
        
        if not self.data_root.exists():
            print(f"Error: VisDrone data not found at {self.data_root}")
            return False
        
        # Process each split
        train_df = self.process_split('train')
        val_df = self.process_split('val')
        test_df = self.process_split('test-dev')
        
        if train_df is None:
            print("\n❌ Failed to process VisDrone")
            return False
        
        print("\n" + "="*70)
        print("VisDrone Processing Complete!")
        if train_df is not None:
            print(f"Train: {len(train_df)} annotations, {train_df['image_id'].nunique()} images")
        if val_df is not None:
            print(f"Val: {len(val_df)} annotations, {val_df['image_id'].nunique()} images")
        if test_df is not None:
            print(f"Test: {len(test_df)} annotations, {test_df['image_id'].nunique()} images")
        print("="*70)
        
        return True


class MOT17Parser:
    """
    Parse MOT17 dataset for tracking evaluation
    
    MOT annotation format (CSV):
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    """
    
    def __init__(self, data_root="data/raw/mot17/MOT17"):
        self.data_root = Path(data_root)
    
    def parse_sequence(self, seq_dir):
        """Parse one MOT17 sequence"""
        gt_file = seq_dir / "gt" / "gt.txt"
        
        if not gt_file.exists():
            return None
        
        # Read ground truth
        df = pd.read_csv(
            gt_file,
            header=None,
            names=['frame', 'track_id', 'x1', 'y1', 'width', 'height', 'conf', 'class', 'visibility']
        )
        
        # Convert to our format
        df['x2'] = df['x1'] + df['width']
        df['y2'] = df['y1'] + df['height']
        df['sequence'] = seq_dir.name
        
        # Read sequence info
        seqinfo_file = seq_dir / "seqinfo.ini"
        if seqinfo_file.exists():
            config = configparser.ConfigParser()
            config.read(seqinfo_file)
            
            df['fps'] = int(config['Sequence']['frameRate'])
            df['img_width'] = int(config['Sequence']['imWidth'])
            df['img_height'] = int(config['Sequence']['imHeight'])
        
        return df
    
    def run(self):
        """Process MOT17 dataset"""
        print("\n" + "="*70)
        print("MOT17 Dataset Preparation")
        print("="*70)
        
        train_dir = self.data_root / "train"
        
        if not train_dir.exists():
            print(f"MOT17 data not found at {self.data_root}")
            print("Skipping (optional for tracking evaluation)")
            return False
        
        # Get all sequences (use only one detector, e.g., FRCNN)
        sequences = sorted([s for s in train_dir.glob('MOT17-*') if 'FRCNN' in s.name])
        
        if len(sequences) == 0:
            print("No MOT17 sequences found")
            return False
        
        print(f"Found {len(sequences)} MOT17 sequences")
        
        all_data = []
        
        for seq_dir in tqdm(sequences, desc="Processing MOT17"):
            df = self.parse_sequence(seq_dir)
            if df is not None:
                all_data.append(df)
                print(f"  {seq_dir.name}: {len(df)} annotations, {df['track_id'].nunique()} tracks")
        
        if len(all_data) == 0:
            return False
        
        # Combine all sequences
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save
        output_dir = Path("data/processed/mot17")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(output_dir / "tracking_annotations.csv", index=False)
        
        # Save metadata
        metadata = {
            'sequences': [seq.name for seq in sequences],
            'total_frames': len(combined_df['frame'].unique()),
            'total_tracks': len(combined_df['track_id'].unique())
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*70)
        print("MOT17 Processing Complete!")
        print(f"Total annotations: {len(combined_df)}")
        print(f"Total sequences: {len(sequences)}")
        print(f"Total tracks: {combined_df['track_id'].nunique()}")
        print("="*70)
        
        return True


def verify_preparation():
    """Verify datasets are properly prepared"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    checks = {}
    
    # Check VisDrone
    train_csv = Path("data/processed/train/annotations.csv")
    
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        
        checks['VisDrone'] = {
            'status': 'OK',
            'train_annotations': len(train_df),
            'train_images': train_df['image_id'].nunique(),
            'classes': train_df['class'].value_counts().to_dict()
        }
        
        # Check condition splits
        conditions = ['day_clear', 'night_clear', 'day_cloudy', 'day_foggy']
        condition_files = [Path(f"data/processed/train/annotations_{c}.csv") for c in conditions]
        checks['VisDrone']['condition_splits'] = sum(1 for f in condition_files if f.exists())
    else:
        checks['VisDrone'] = {'status': 'MISSING'}
    
    # Check MOT17
    mot17_csv = Path("data/processed/mot17/tracking_annotations.csv")
    
    if mot17_csv.exists():
        mot17_df = pd.read_csv(mot17_csv)
        
        checks['MOT17'] = {
            'status': 'OK',
            'sequences': mot17_df['sequence'].nunique(),
            'tracks': mot17_df['track_id'].nunique(),
            'frames': mot17_df['frame'].nunique()
        }
    else:
        checks['MOT17'] = {'status': 'MISSING'}
    
    # Print results
    print("\nDataset Preparation Status:")
    print("-"*70)
    
    for dataset, info in checks.items():
        status = info['status']
        icon = "✅" if status == 'OK' else "❌"
        
        print(f"\n{icon} {dataset}: {status}")
        
        if status == 'OK':
            for key, value in info.items():
                if key != 'status':
                    print(f"    {key}: {value}")
    
    return checks


def create_yolo_format():
    """Convert VisDrone to YOLO format for training"""
    print("\n" + "="*70)
    print("Creating YOLO Format Dataset")
    print("="*70)
    
    # Read processed annotations
    train_csv = Path("data/processed/train/annotations.csv")
    val_csv = Path("data/processed/val/annotations.csv")
    
    if not train_csv.exists():
        print("Error: Processed VisDrone data not found")
        return False
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv) if val_csv.exists() else None
    
    # Create YOLO directories
    yolo_root = Path("data/processed/yolo")
    
    for split, df in [('train', train_df), ('val', val_df)]:
        if df is None:
            continue
        
        images_dir = yolo_root / split / 'images'
        labels_dir = yolo_root / split / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split} split for YOLO format...")
        
        # Group by image
        for image_id, group in tqdm(df.groupby('image_id'), desc=f"Converting {split}"):
            # Get image path
            img_rel_path = group.iloc[0]['image_path']
            img_path = Path("data/raw/visdrone") / img_rel_path
            
            if not img_path.exists():
                continue
            
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Copy or symlink image
            out_img = images_dir / f"{image_id}.jpg"
            if not out_img.exists():
                # Create symlink or copy
                try:
                    out_img.symlink_to(img_path.absolute())
                except:
                    import shutil
                    shutil.copy(img_path, out_img)
            
            # Create YOLO label file
            label_lines = []
            for _, row in group.iterrows():
                # Convert to YOLO format (normalized xywh)
                x_center = ((row['x1'] + row['x2']) / 2) / w
                y_center = ((row['y1'] + row['y2']) / 2) / h
                box_width = (row['x2'] - row['x1']) / w
                box_height = (row['y2'] - row['y1']) / h
                
                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))
                
                class_id = row['class_id']
                
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            
            # Save label file
            label_file = labels_dir / f"{image_id}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))
        
        print(f"  Created {len(list(labels_dir.glob('*.txt')))} label files")
    
    # Create data.yaml
    yaml_content = f"""# VisDrone dataset for YOLOv8
path: {yolo_root.absolute()}
train: train/images
val: val/images

# Classes
nc: 5
names: ['car', 'van', 'bus', 'truck', 'others']
"""
    
    yaml_path = yolo_root / "visdrone.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ YOLO dataset configuration saved to: {yaml_path}")
    
    return True


def print_next_steps(checks):
    """Print next steps based on verification"""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if checks.get('VisDrone', {}).get('status') == 'OK':
        print("\n✅ VisDrone is ready for training!")
        print("\nYou can now:")
        print("1. Train detection model:")
        print("   python src/detection/train_yolov8.py")
        print("      (will use: data/processed/yolo/visdrone.yaml)")
        print("\n2. Or run the full pipeline:")
        print("   python src/pipeline.py --video <path> --output result.mp4")
    else:
        print("\n❌ VisDrone preparation failed")
        print("Please check that:")
        print("  1. Dataset is downloaded to data/raw/visdrone/")
        print("  2. Run: python scripts/download_datasets_v2.py")
    
    if checks.get('MOT17', {}).get('status') == 'OK':
        print("\n✅ MOT17 is ready for tracking evaluation!")
    else:
        print("\n⚠️ MOT17 not prepared (optional)")
        print("Tracking evaluation will use VisDrone data")


def print_dataset_summary():
    """Print summary of prepared datasets"""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    # VisDrone summary
    train_csv = Path("data/processed/train/annotations.csv")
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        
        print("\n📊 VisDrone Dataset:")
        print(f"  Total images: {train_df['image_id'].nunique()}")
        print(f"  Total vehicles: {len(train_df)}")
        print(f"\n  Vehicle distribution:")
        for cls, count in train_df['class'].value_counts().items():
            print(f"    {cls}: {count}")
        
        print(f"\n  Condition distribution:")
        if 'illumination' in train_df.columns:
            print(f"    Day: {sum(train_df['illumination'] == 'day')}")
            print(f"    Night: {sum(train_df['illumination'] == 'night')}")
        if 'weather' in train_df.columns:
            for weather, count in train_df['weather'].value_counts().items():
                print(f"    {weather}: {count}")
    
    # MOT17 summary
    mot17_csv = Path("data/processed/mot17/tracking_annotations.csv")
    if mot17_csv.exists():
        mot17_df = pd.read_csv(mot17_csv)
        
        print("\n📊 MOT17 Dataset:")
        print(f"  Sequences: {mot17_df['sequence'].nunique()}")
        print(f"  Total frames: {mot17_df['frame'].nunique()}")
        print(f"  Total tracks: {mot17_df['track_id'].nunique()}")
        print(f"  Annotations: {len(mot17_df)}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" "*15 + "DATASET PREPARATION - PHASE 1 (v2)")
    print("="*70)
    
    print("\nParsing VisDrone and MOT17 datasets...")
    
    success_count = 0
    
    # Parse VisDrone
    if Path("data/raw/visdrone").exists():
        parser = VisDroneParser()
        if parser.run():
            success_count += 1
    else:
        print("\n⚠ VisDrone not found at data/raw/visdrone/")
        print("Please run: python scripts/download_datasets_v2.py")
    
    # Parse MOT17
    if Path("data/raw/mot17/MOT17").exists():
        parser = MOT17Parser()
        if parser.run():
            success_count += 1
    else:
        print("\n⚠ MOT17 not found (optional)")
    
    # Create YOLO format
    if success_count > 0:
        create_yolo_format()
    
    # Verify everything
    checks = verify_preparation()
    
    # Print summary
    print_dataset_summary()
    
    # Print next steps
    print_next_steps(checks)
    
    print("\n" + "="*70)
    print("PREPARATION SUMMARY")
    print("="*70)
    print(f"\nDatasets successfully prepared: {success_count}")
    
    if checks.get('VisDrone', {}).get('status') == 'OK':
        print("\n🎉 SUCCESS! Ready to start training.")
    else:
        print("\n⚠ Preparation incomplete. Check errors above.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()