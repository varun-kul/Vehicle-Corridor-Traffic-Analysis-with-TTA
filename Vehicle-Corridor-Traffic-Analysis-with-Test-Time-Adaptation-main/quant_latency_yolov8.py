#!/usr/bin/env python3
"""
quant_latency_yolov8.py
 
Small experiment to compare inference latency for:
- GPU FP32
- CPU FP32
- CPU dynamic quantized INT8 (Linear layers)
 
Usage example (PowerShell):
  python quant_latency_yolov8.py `
    --weights outputs/detection/yolov8n_visdrone3/weights/best.pt `
    --imgsz 640 `
    --runs 50 `
    --warmup 10 `
    --out outputs/metrics/quant_latency.json
"""
 
import argparse
import json
import time
import copy
from pathlib import Path
 
import torch
from ultralytics import YOLO
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure YOLOv8 latency on GPU/CPU FP32 and CPU INT8 (dynamic)."
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained YOLO weights (e.g. best.pt).",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Image size (square).")
    p.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed inference runs per config.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs (not timed).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs/metrics/quant_latency.json",
        help="Path to save JSON results.",
    )
    return p.parse_args()
 
 
def benchmark_module(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int,
    warmup: int,
    runs: int,
) -> dict:
    model.eval()
    model.to(device)
 
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
 
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
 
    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
 
    return {
        "device": str(device),
        "runs": runs,
        "warmup": warmup,
        "mean_ms": float(sum(times_ms) / len(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }
 
 
def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    print(f"Loading model from {args.weights} ...")
    yolo = YOLO(args.weights)
    base_module = yolo.model  # underlying nn.Module
 
    results = {"imgsz": args.imgsz, "runs": args.runs, "warmup": args.warmup, "configs": {}}
 
    # ---------------- GPU FP32 (if available) ----------------
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda:0")
        print("\n[1] Benchmarking GPU FP32 ...")
        mod_gpu = copy.deepcopy(base_module)
        res_gpu = benchmark_module(mod_gpu, device_gpu, args.imgsz, args.warmup, args.runs)
        results["configs"]["gpu_fp32"] = res_gpu
        print(
            f"GPU FP32: mean={res_gpu['mean_ms']:.2f} ms, "
            f"min={res_gpu['min_ms']:.2f}, max={res_gpu['max_ms']:.2f}"
        )
    else:
        print("\n[1] GPU not available; skipping GPU FP32 benchmark.")
 
    # ---------------- CPU FP32 ----------------
    device_cpu = torch.device("cpu")
    print("\n[2] Benchmarking CPU FP32 ...")
    mod_cpu_fp32 = copy.deepcopy(base_module)
    res_cpu_fp32 = benchmark_module(mod_cpu_fp32, device_cpu, args.imgsz, args.warmup, args.runs)
    results["configs"]["cpu_fp32"] = res_cpu_fp32
    print(
        f"CPU FP32: mean={res_cpu_fp32['mean_ms']:.2f} ms, "
        f"min={res_cpu_fp32['min_ms']:.2f}, max={res_cpu_fp32['max_ms']:.2f}"
    )
 
    # ---------------- CPU INT8 dynamic quantization ----------------
    print("\n[3] Benchmarking CPU dynamic INT8 (Linear layers) ...")
    try:
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:
            from torch.quantization import quantize_dynamic  # older PyTorch
 
        mod_cpu_int8 = copy.deepcopy(base_module).to("cpu")
        # Only Linear layers are quantized; Conv layers stay FP32 (safe and simple).
        mod_cpu_int8 = quantize_dynamic(mod_cpu_int8, {torch.nn.Linear}, dtype=torch.qint8)
 
        res_cpu_int8 = benchmark_module(
            mod_cpu_int8, device_cpu, args.imgsz, args.warmup, args.runs
        )
        results["configs"]["cpu_int8_dynamic"] = res_cpu_int8
        print(
            f"CPU INT8 dynamic: mean={res_cpu_int8['mean_ms']:.2f} ms, "
            f"min={res_cpu_int8['min_ms']:.2f}, max={res_cpu_int8['max_ms']:.2f}"
        )
    except Exception as e:
        print(f"Warning: dynamic quantization failed: {e}")
        results["configs"]["cpu_int8_dynamic"] = {"error": str(e)}
 
    # ---------------- Save JSON ----------------
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
 
    print(f"\nLatency experiment saved to {out_path.resolve()}")
 
 
if __name__ == "__main__":
    main()