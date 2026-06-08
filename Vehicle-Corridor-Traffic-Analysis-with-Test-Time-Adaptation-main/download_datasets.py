#!/usr/bin/env python3
"""
scripts/download_datasets_v2.py
Download VisDrone and MOT17 datasets (readily available)
"""

import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import sys
import time

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, desc="Downloading"):
    """Download file with progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{desc}: {output_path.name}")
    print(f"URL: {url}")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file with progress"""
    print(f"\nExtracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False

def download_visdrone():
    """
    VisDrone Dataset - Drone-based traffic monitoring
    Perfect for vehicle detection, tracking, and counting
    
    Dataset Info:
    - 10,209 images with vehicle annotations
    - Multiple weather conditions (sunny, foggy, rainy)
    - Day and night scenes
    - Various crowd densities
    - Vehicle classes: car, van, bus, truck
    
    Paper: https://arxiv.org/abs/1804.07437
    """
    print("\n" + "="*70)
    print("VISDRONE DATASET")
    print("="*70)
    
    base_url = "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0"
    output_dir = Path("data/raw/visdrone")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset splits
    datasets = {
        "VisDrone2019-DET-train.zip": "Training set (6,471 images)",
        "VisDrone2019-DET-val.zip": "Validation set (548 images)",
        "VisDrone2019-DET-test-dev.zip": "Test set (1,610 images)"
    }
    
    print("\nVisDrone includes:")
    print("- Detection annotations for vehicles")
    print("- Multiple weather/lighting conditions")
    print("- Perfect for traffic monitoring")
    print(f"\nTotal size: ~2.5 GB")
    
    downloaded = []
    
    for filename, description in datasets.items():
        print(f"\n--- {description} ---")
        
        zip_path = output_dir / filename
        
        # Check if already exists
        extracted_dir = output_dir / filename.replace('.zip', '')
        if extracted_dir.exists() and len(list(extracted_dir.glob('*'))) > 0:
            print(f"✓ Already exists: {filename}")
            downloaded.append(filename)
            continue
        
        # Download if needed
        if not zip_path.exists():
            url = f"{base_url}/{filename}"
            success = download_file(url, zip_path, f"Downloading {filename}")
            
            if not success:
                print(f"⚠ Failed to download {filename}")
                print(f"You can manually download from:")
                print(f"  {url}")
                print(f"  And place in: {output_dir}")
                continue
        
        # Extract
        if zip_path.exists():
            success = extract_zip(zip_path, output_dir)
            if success:
                downloaded.append(filename)
                # Clean up zip file to save space
                # zip_path.unlink()
    
    # Verify structure
    train_dir = output_dir / "VisDrone2019-DET-train"
    if train_dir.exists():
        images = list((train_dir / "images").glob("*.jpg"))
        annotations = list((train_dir / "annotations").glob("*.txt"))
        
        print(f"\n✓ VisDrone training set ready:")
        print(f"  Images: {len(images)}")
        print(f"  Annotations: {len(annotations)}")
        
        return True
    else:
        print(f"\n⚠ VisDrone structure incomplete")
        return False

def download_mot17():
    """
    MOT17 Dataset - Multiple Object Tracking benchmark
    Excellent for tracking evaluation
    
    Dataset Info:
    - 14 video sequences (7 train, 7 test)
    - Ground truth tracking annotations
    - Multiple detection methods included
    - Pedestrian and vehicle tracking
    - Indoor and outdoor scenes
    
    Website: https://motchallenge.net/
    """
    print("\n" + "="*70)
    print("MOT17 DATASET")
    print("="*70)
    
    output_dir = Path("data/raw/mot17")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MOT17 download URL
    url = "https://motchallenge.net/data/MOT17.zip"
    zip_path = output_dir / "MOT17.zip"
    
    print("\nMOT17 includes:")
    print("- 7 training sequences with ground truth")
    print("- Multiple detection methods (DPM, FRCNN, SDP)")
    print("- Perfect for tracking evaluation")
    print(f"\nTotal size: ~5.5 GB")
    
    # Check if already extracted
    train_dir = output_dir / "MOT17" / "train"
    if train_dir.exists() and len(list(train_dir.glob('MOT17-*'))) > 0:
        sequences = list(train_dir.glob('MOT17-*'))
        print(f"\n✓ MOT17 already exists with {len(sequences)} sequences")
        return True
    
    # Download
    if not zip_path.exists():
        print("\nAttempting download from motchallenge.net...")
        success = download_file(url, zip_path, "Downloading MOT17")
        
        if not success:
            print("\n⚠ Automatic download failed")
            print("\nManual download instructions:")
            print("1. Visit: https://motchallenge.net/data/MOT17/")
            print("2. Download MOT17.zip")
            print(f"3. Place in: {output_dir}")
            print("4. Re-run this script")
            return False
    
    # Extract
    if zip_path.exists():
        print(f"\nExtracting MOT17 (this may take a while)...")
        success = extract_zip(zip_path, output_dir)
        
        if success:
            # Verify
            train_dir = output_dir / "MOT17" / "train"
            if train_dir.exists():
                sequences = list(train_dir.glob('MOT17-*'))
                print(f"\n✓ MOT17 ready with {len(sequences)} training sequences")
                
                # List sequences
                for seq in sorted(sequences)[:5]:
                    img_dir = seq / "img1"
                    if img_dir.exists():
                        num_imgs = len(list(img_dir.glob('*.jpg')))
                        print(f"  {seq.name}: {num_imgs} frames")
                
                return True
        
    return False

def download_coco_vehicles():
    """
    Instructions for COCO dataset (optional warm-start)
    Can be auto-downloaded by most frameworks
    """
    print("\n" + "="*70)
    print("COCO DATASET (Optional)")
    print("="*70)
    
    print("\nCOCO can be used for:")
    print("- Pre-training on diverse vehicle images")
    print("- Warm-start before fine-tuning on VisDrone")
    print("- Classes: car, bus, truck, motorcycle")
    
    print("\nCOCO will auto-download when using YOLOv8:")
    print("  model.train(data='coco.yaml', ...)")
    
    print("\nOr manually download:")
    print("1. Visit: https://cocodataset.org/")
    print("2. Download train2017 and val2017")
    print("3. Download annotations")
    
    return True

def verify_downloads():
    """Verify all datasets are properly downloaded"""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    results = {}
    
    # Check VisDrone
    visdrone_train = Path("data/raw/visdrone/VisDrone2019-DET-train")
    if visdrone_train.exists():
        images = list((visdrone_train / "images").glob("*.jpg"))
        results['VisDrone'] = {
            'status': 'OK',
            'train_images': len(images)
        }
    else:
        results['VisDrone'] = {'status': 'MISSING'}
    
    # Check MOT17
    mot17_train = Path("data/raw/mot17/MOT17/train")
    if mot17_train.exists():
        sequences = list(mot17_train.glob('MOT17-*'))
        results['MOT17'] = {
            'status': 'OK',
            'sequences': len(sequences)
        }
    else:
        results['MOT17'] = {'status': 'MISSING'}
    
    # Print results
    print("\nDataset Status:")
    for dataset, info in results.items():
        status = info['status']
        icon = "✅" if status == 'OK' else "❌"
        
        print(f"\n{icon} {dataset}: {status}")
        if status == 'OK':
            for key, value in info.items():
                if key != 'status':
                    print(f"    {key}: {value}")
    
    # Overall status
    all_ok = all(info['status'] == 'OK' for info in results.values())
    
    if all_ok:
        print("\n" + "="*70)
        print("✅ ALL DATASETS READY!")
        print("="*70)
        print("\nNext step:")
        print("  python scripts/prepare_datasets_v2.py")
    else:
        print("\n" + "="*70)
        print("⚠ SOME DATASETS MISSING")
        print("="*70)
        print("\nPlease check error messages above")
        print("You may need to download manually")
    
    return results

def print_dataset_info():
    """Print information about the datasets"""
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    
    print("\n📊 VisDrone (Primary - Detection & Tracking)")
    print("  Source: Drone-based traffic surveillance")
    print("  Size: ~2.5 GB")
    print("  Images: 10,209")
    print("  Use for: Detection training, tracking, day/night/weather testing")
    print("  Classes: pedestrian, car, van, bus, truck, etc.")
    
    print("\n📊 MOT17 (Tracking Evaluation)")
    print("  Source: Multi-object tracking benchmark")
    print("  Size: ~5.5 GB")
    print("  Sequences: 14 (7 train, 7 test)")
    print("  Use for: Tracking metrics (MOTA, IDF1, ID switches)")
    print("  Provides: Ground truth tracks for evaluation")
    
    print("\n📊 Total Download: ~8 GB")

def main():
    """Main execution"""
    print("="*70)
    print(" "*15 + "DATASET DOWNLOAD - PHASE 1 (v2)")
    print("="*70)
    print("\nUsing VisDrone + MOT17 (readily available datasets)")
    
    # Print information
    print_dataset_info()
    
    print("\n" + "="*70)
    print("STARTING DOWNLOADS")
    print("="*70)
    
    # Create base directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    visdrone_ok = download_visdrone()
    mot17_ok = download_mot17()
    download_coco_vehicles()
    
    # Verify
    results = verify_downloads()
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    if visdrone_ok and mot17_ok:
        print("\n🎉 SUCCESS! All required datasets are ready.")
        print("\n📋 What you have:")
        print("  ✓ VisDrone - for detection and tracking")
        print("  ✓ MOT17 - for tracking evaluation")
        
        print("\n🚀 Next steps:")
        print("  1. Prepare datasets:")
        print("     python scripts/prepare_datasets_v2.py")
        print("\n  2. Train detection model:")
        print("     python src/detection/train_yolov8.py")
        print("\n  3. Run full pipeline:")
        print("     python src/pipeline.py --video <path>")
    else:
        print("\n⚠ Some downloads incomplete")
        print("Please check error messages and retry")
        print("\nYou can also manually download:")
        print("  VisDrone: https://github.com/VisDrone/VisDrone-Dataset")
        print("  MOT17: https://motchallenge.net/data/MOT17/")

if __name__ == "__main__":
    main()