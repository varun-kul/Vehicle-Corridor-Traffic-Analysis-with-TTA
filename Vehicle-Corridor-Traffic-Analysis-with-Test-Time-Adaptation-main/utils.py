
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from typing import List

def is_video_path(p: str) -> bool:
    p = str(p).lower()
    return p.endswith((".mp4",".avi",".mov",".mkv",".webm")) or p.isdigit()

def list_images(root: str) -> List[str]:
    root = Path(root)
    exts = [".jpg",".jpeg",".png",".bmp"]
    files = []
    if root.is_dir():
        for e in exts:
            files.extend(sorted([str(p) for p in root.glob(f"*{e}")])) 
    elif root.is_file():
        files = [str(root)]
    return files

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), stride=32, auto=True, scaleFill=False, scaleup=True):
    # Simplified letterbox (based on Ultralytics)
    shape = img.shape[:2]  # H, W
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def frames_from_source(source: str, every_n: int = 1, limit: int | None = None):
    """Yield BGR frames from a video (or camera index) or a directory of images."""
    if is_video_path(source):
        cap = cv2.VideoCapture(0 if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        idx = -1
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % every_n != 0:
                continue
            yield frame
            yielded += 1
            if limit is not None and yielded >= limit:
                break
        cap.release()
    else:
        for p in list_images(source):
            frame = cv2.imread(p)
            if frame is None:
                continue
            yield frame
