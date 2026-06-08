# Vehicle Corridor Traffic Analysis with Test Time Adaptation

This repository contains the code for **Vehicle Corridor Traffic Analysis with Test Time Adaptation**.  
The project builds a two phase, edge oriented pipeline for:

- multi class vehicle detection on VisDrone using YOLOv8n
- multi object tracking on VisDrone MOT and MOT17 using ByteTrack
- approximate per vehicle speed estimation from tracks
- simple test time adaptation (AdaBN, Tent, pseudo labels)
- a small post training quantization and latency study on GPU and CPU

The full technical details and results are in the report  
`Vehicle Corridor Traffic Analysis With Test - Time Domain Adaptation.pdf`.

---

## 1. High level overview

**Phase 1 - Baseline pipeline**

1. Prepare VisDrone and MOT17 datasets.
2. Fine tune a compact YOLOv8n detector on VisDrone vehicle labels  
   (car, van, bus, truck, others).
3. Evaluate detection on VisDrone validation (mAP, precision, recall).
4. Run ByteTrack on VisDrone MOT and MOT17 to obtain tracks.
5. Convert tracks to approximate speeds in km/h using a single meters per pixel scale.
6. Export annotated videos and CSVs with `frame, id, class, speed`.

**Phase 2 - Robustness and deployment**

1. Add unsupervised test time adaptation (TTA) on BatchNorm layers  
   (AdaBN, Tent style entropy, pseudo label adaptation).
2. Compare detection metrics with and without TTA.
3. Run YOLOv8n on GPU and CPU and apply dynamic INT8 quantization to linear layers.
4. Measure latency for GPU FP32, CPU FP32, and CPU INT8 dynamic.

---

## 2. Main scripts

- `download_datasets.py` - download or verify VisDrone and MOT17 into `data/raw`.
- `prepare_datasets.py` - build YOLO format VisDrone data and MOT style folders.
- `test_yolov8.py` - train or fine tune YOLOv8n on VisDrone.
- `validate_yolov8.py` - run validation on VisDrone and write JSON metrics.
- `test_detection.py` - quick detection demo on sample images or video.
- `test_tracking.py` - run ByteTrack tracking and save MOT style CSVs and videos.
- `evaluate_baseline.py` - baseline MOT17 tracking metrics (IDF1, MOTA etc.) if motmetrics is available.
- `speed_from_mot.py` - speed estimation from baseline tracks.
- `adapters.py` - TTA adapters (AdaBN, Tent entropy, pseudo labels).
- `eval_baseline_vs_tta_detection.py` - detection baseline vs TTA comparison on VisDrone.
- `run_tta_eval_visdrone.py` - wrapper to run VisDrone validation with a chosen TTA mode.
- `run_tta_track.py` - run YOLO + TTA + ByteTrack on a MOT sequence.
- `speed_from_mot_tta.py` - speed estimation from TTA tracks.
- `quant_latency_yolov8.py` - quantization and latency benchmark.
- `convert_mot_to_mp4.py` - convert MOT `img1` frame folders to MP4.
- `interactive_calibrate_homography.py` - optional interactive homography tool (not required for basic runs).
- `run_all_results.py` - convenience script to execute the main experiments in sequence.
- `utils.py`, `master.py` - shared utilities and playground script.

---
