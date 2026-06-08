#!/usr/bin/env python3
from __future__ import annotations
import argparse, configparser, re
from pathlib import Path
import cv2

def natural_key(p: Path):
    # 000001.jpg → 1, ensures correct numeric ordering
    s = p.stem
    m = re.search(r'(\d+)$', s)
    return int(m.group(1)) if m else s

def read_seqinfo(seq_root: Path):
    ini = seq_root / "seqinfo.ini"
    if not ini.exists():
        raise SystemExit(f"Missing seqinfo.ini in {seq_root}")
    cfg = configparser.ConfigParser()
    cfg.read(ini.as_posix())
    S = cfg["Sequence"]
    info = {
        "name": S.get("name", seq_root.name),
        "imDir": S.get("imDir", "img1"),
        "frameRate": S.getint("frameRate", 30),
        "seqLength": S.getint("seqLength", 0),
        "imWidth": S.getint("imWidth", 0),
        "imHeight": S.getint("imHeight", 0),
        "imExt": S.get("imExt", ".jpg"),
    }
    return info

def find_frames_dir(seq_root: Path, imDir_hint: str):
    # prefer imDir from ini; otherwise common fallbacks
    candidates = [imDir_hint, "images", "img1", "img"]
    for c in candidates:
        d = seq_root / c
        if d.exists() and d.is_dir():
            return d
    raise SystemExit(f"No frames directory found under {seq_root} (tried {candidates})")

def collect_frames(frames_dir: Path, imExt: str):
    ext = imExt if imExt.startswith(".") else "." + imExt
    files = sorted(frames_dir.glob(f"*{ext}"), key=natural_key)
    if not files:
        # try case-insensitive / any extension
        files = sorted([p for p in frames_dir.iterdir() if p.is_file()],
                       key=natural_key)
    if not files:
        raise SystemExit(f"No frame images found in {frames_dir}")
    return files

def infer_size_from_first(frame_path: Path, fallback_w: int, fallback_h: int):
    img = cv2.imread(frame_path.as_posix())
    if img is None:
        raise SystemExit(f"Cannot read first frame: {frame_path}")
    h, w = img.shape[:2]
    if fallback_w and fallback_h:
        return fallback_w, fallback_h
    return w, h

def write_mp4(files, out_path: Path, fps: float, size_wh: tuple[int, int]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path.as_posix(), fourcc, fps, size_wh)
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open. Try a different --fourcc or path.")
    for i, f in enumerate(files, 1):
        img = cv2.imread(f.as_posix())
        if img is None:
            print(f"[warn] Skipping unreadable frame: {f}")
            continue
        if (img.shape[1], img.shape[0]) != size_wh:
            img = cv2.resize(img, size_wh)
        writer.write(img)
    writer.release()

def convert_sequence(seq_root: Path, out_dir: Path, overwrite: bool):
    info = read_seqinfo(seq_root)
    frames_dir = find_frames_dir(seq_root, info["imDir"])
    files = collect_frames(frames_dir, info["imExt"])
    w, h = infer_size_from_first(files[0], info["imWidth"], info["imHeight"])
    fps = info["frameRate"]

    out_path = out_dir / f"{info['name']}.mp4"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists. Use --overwrite to regenerate.")
        return out_path, fps

    print(f"[convert] {seq_root.name}: {len(files)} frames @ {fps} fps → {out_path.name} ({w}x{h})")
    write_mp4(files, out_path, fps, (w, h))
    print(f"[done] wrote {out_path}")
    return out_path, fps

def main():
    ap = argparse.ArgumentParser(description="Convert MOT17 sequence(s) to MP4 using seqinfo.ini")
    ap.add_argument("--root", required=True,
                    help="Path to a sequence folder containing seqinfo.ini OR a directory with many sequences")
    ap.add_argument("--out_dir", default="outputs/videos")
    ap.add_argument("--recursive", action="store_true",
                    help="If --root has many sequences, process all subfolders containing seqinfo.ini")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    if args.recursive:
        seqs = [p for p in root.rglob("seqinfo.ini")]
        if not seqs:
            raise SystemExit("No seqinfo.ini found under --root")
        for ini in seqs:
            convert_sequence(ini.parent, out_dir, args.overwrite)
    else:
        convert_sequence(root, out_dir, args.overwrite)

if __name__ == "__main__":
    main()
