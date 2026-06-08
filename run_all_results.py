#!/usr/bin/env python3
"""
run_all_results.py
 
Runs the main evaluation scripts for the Vehicle Corridor Traffic Analysis project
and prints a concise summary of the key results.
 
It will use GPU if available (device=cuda), otherwise CPU (device=cpu).
It assumes YOLO weights already exist at WEIGHTS_PATH.
"""
 
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
 
# Try to detect CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False
 
DEVICE_ARG = "cuda" if HAS_CUDA else "cpu"
 
# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
 
WEIGHTS_PATH = Path("outputs/detection/yolov8n_visdrone3/weights/best.pt")
DATA_YAML = Path("data/processed/yolo/visdrone.yaml")
 
VAL_METRICS_JSON = Path("outputs/metrics/val_metrics.json")
TTA_METRICS_JSON = Path("outputs/metrics/baseline_vs_tta_visdrone.json")
QUANT_JSON = Path("outputs/metrics/quant_latency.json")
 
# Script locations
VALIDATE_SCRIPT = Path("scripts/validate_yolov8.py")
EVAL_TTA_DET_SCRIPT = Path("eval_baseline_vs_tta_detection.py")
QUANT_SCRIPT = Path("quant_latency_yolov8.py")
EVAL_MOT_SCRIPT = Path("scripts/evaluate_baseline.py")
 
 
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
 
def run_command(title: str, cmd: list[str]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)
    print("Command:")
    print("  " + " ".join(cmd))
    print("-" * 70)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] command exited with code {result.returncode}")
    else:
        print("[OK] command finished")
 
 
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] JSON file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
 
 
# ---------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------
 
def print_detection_summary() -> None:
    data = load_json(VAL_METRICS_JSON)
    if data is None:
        return
 
    # validate_yolov8.py stores payload as {"metrics": {...}}
    metrics = data.get("metrics", data)
    summary = metrics.get("summary", metrics)
 
    print("\nDetection summary on VisDrone (validate_yolov8):")
    print(f"  mAP50-95: {summary.get('map50_95', 0.0):.4f}")
    print(f"  mAP50:    {summary.get('map50', 0.0):.4f}")
    print(f"  mp:       {summary.get('mp', 0.0):.4f}")
    print(f"  mr:       {summary.get('mr', 0.0):.4f}")
 
    speed_ms = summary.get("speed_ms", {})
    if speed_ms:
        print("  Speed breakdown [ms per image]:")
        for k, v in speed_ms.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            print(f"    {k}: {val:.3f}")
 
    per_class = summary.get("per_class_ap50") or metrics.get("per_class_ap50")
    if isinstance(per_class, dict):
        print("  Class-wise AP50:")
        for cname, ap in per_class.items():
            try:
                val = float(ap)
            except (TypeError, ValueError):
                continue
            print(f"    {cname}: {val:.3f}")
 
 
def print_tta_detection_summary() -> None:
    data = load_json(TTA_METRICS_JSON)
    if data is None:
        return
 
    base = data.get("baseline", {})
    tta_block = data.get("tta", {})
    tta_metrics = tta_block.get("metrics", {})
    mode = tta_block.get("mode", "unknown")
 
    print("\nBaseline vs TTA detection on VisDrone (eval_baseline_vs_tta_detection):")
    print(f"  TTA mode: {mode}")
    print("  Baseline:")
    print(f"    mAP50-95: {base.get('map50_95', 0.0):.4f}")
    print(f"    mAP50:    {base.get('map50', 0.0):.4f}")
    print(f"    mp:       {base.get('mp', 0.0):.4f}")
    print(f"    mr:       {base.get('mr', 0.0):.4f}")
 
    print("  With TTA:")
    print(f"    mAP50-95: {tta_metrics.get('map50_95', 0.0):.4f}")
    print(f"    mAP50:    {tta_metrics.get('map50', 0.0):.4f}")
    print(f"    mp:       {tta_metrics.get('mp', 0.0):.4f}")
    print(f"    mr:       {tta_metrics.get('mr', 0.0):.4f}")
 
    dm = tta_metrics.get("map50_95", 0.0) - base.get("map50_95", 0.0)
    d50 = tta_metrics.get("map50", 0.0) - base.get("map50", 0.0)
    dmp = tta_metrics.get("mp", 0.0) - base.get("mp", 0.0)
    dmr = tta_metrics.get("mr", 0.0) - base.get("mr", 0.0)
 
    print("  Differences (TTA minus baseline):")
    print(f"    Δ mAP50-95: {dm:+.4f}")
    print(f"    Δ mAP50:    {d50:+.4f}")
    print(f"    Δ mp:       {dmp:+.4f}")
    print(f"    Δ mr:       {dmr:+.4f}")
 
 
def print_quant_summary() -> None:
    data = load_json(QUANT_JSON)
    if data is None:
        return
    cfgs = data.get("configs", {})
    gpu = cfgs.get("gpu_fp32")
    cpu = cfgs.get("cpu_fp32")
    int8 = cfgs.get("cpu_int8_dynamic")
 
    print("\nQuantization and latency summary (quant_latency_yolov8):")
    if gpu:
        print(f"  GPU FP32 mean latency: {gpu['mean_ms']:.2f} ms")
    if cpu:
        print(f"  CPU FP32 mean latency: {cpu['mean_ms']:.2f} ms")
    if int8 and "mean_ms" in int8:
        print(f"  CPU INT8 mean latency: {int8['mean_ms']:.2f} ms")
 
    if gpu and cpu:
        ratio = cpu["mean_ms"] / gpu["mean_ms"]
        print(f"  CPU FP32 vs GPU FP32: {ratio:.2f}x slower")
    if cpu and int8 and "mean_ms" in int8:
        speedup = cpu["mean_ms"] / int8["mean_ms"]
        print(
            f"  CPU INT8 vs CPU FP32: {speedup:.2f}x "
            f"({'about %.1f%% speed-up' % ((speedup - 1.0) * 100)})"
        )
    if gpu and int8 and "mean_ms" in int8:
        ratio = int8["mean_ms"] / gpu["mean_ms"]
        print(f"  CPU INT8 vs GPU FP32: {ratio:.2f}x slower")
 
 
# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------
 
def main() -> None:
    print("=" * 70)
    print("Vehicle Corridor Traffic Analysis - results runner")
    print("=" * 70)
    print(f"[INFO] Using device argument: {DEVICE_ARG}")
    print("=" * 70)
 
    # 1) Detection validation
    if VALIDATE_SCRIPT.exists():
        cmd = [
            "python",
            str(VALIDATE_SCRIPT),
            "--weights",
            str(WEIGHTS_PATH),
            "--data",
            str(DATA_YAML),
            "--imgsz",
            "640",
            "--device",
            DEVICE_ARG,
            "--out",
            str(VAL_METRICS_JSON),
        ]
        run_command("Detection validation on VisDrone", cmd)
    else:
        print(f"[WARN] {VALIDATE_SCRIPT} not found, skipping detection validation")
 
    print_detection_summary()
 
    # 2) Baseline vs TTA detection
    if EVAL_TTA_DET_SCRIPT.exists():
        cmd = [
            "python",
            str(EVAL_TTA_DET_SCRIPT),
            "--weights",
            str(WEIGHTS_PATH),
            "--data",
            str(DATA_YAML),
            "--imgsz",
            "640",
            "--device",
            DEVICE_ARG,
            "--tta-mode",
            "adabn",
            "--tta-batches",
            "128",
            "--tta-steps",
            "1",
            "--tta-lr",
            "1e-4",
            "--tta-conf-thr",
            "0.7",
            "--out",
            str(TTA_METRICS_JSON),
        ]
        run_command("Baseline vs AdaBN TTA detection on VisDrone", cmd)
    else:
        print(f"[WARN] {EVAL_TTA_DET_SCRIPT} not found, skipping TTA detection evaluation")
 
    print_tta_detection_summary()
 
    # 3) MOT evaluation (optional; may fail if tracking package is missing)
    if EVAL_MOT_SCRIPT.exists():
        cmd = ["python", str(EVAL_MOT_SCRIPT)]
        run_command("MOT17 tracking evaluation (baseline)", cmd)
    else:
        print(f"[WARN] {EVAL_MOT_SCRIPT} not found, skipping MOT evaluation")
 
    # 4) Quantization and latency experiment
    if QUANT_SCRIPT.exists():
        cmd = [
            "python",
            str(QUANT_SCRIPT),
            "--weights",
            str(WEIGHTS_PATH),
            "--imgsz",
            "640",
            "--runs",
            "50",
            "--warmup",
            "10",
            "--out",
            str(QUANT_JSON),
        ]
        run_command("Quantization and latency experiment", cmd)
    else:
        print(f"[WARN] {QUANT_SCRIPT} not found, skipping quantization experiment")
 
    print_quant_summary()
 
    print("\n" + "=" * 70)
    print("Done. Detection, TTA, MOT (if available), and quantization results")
    print("have been run and summarised above.")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()