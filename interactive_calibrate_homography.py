# interactive_calibrate_homography.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Interactively pick image points on the road plane and enter "
            "their world coordinates (meters) to compute a homography."
        )
    )
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a video file or a single image frame.",
    )
    p.add_argument(
        "--out-h",
        type=str,
        default="homography.json",
        help="Output JSON file to store 3x3 homography matrix H.",
    )
    p.add_argument(
        "--out-points",
        type=str,
        default="calib_points.json",
        help="Output JSON file to store image/world point pairs.",
    )
    return p.parse_args()


def read_first_frame(path: str):
    # allow both image and video
    if any(path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return img

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame from video.")
    return frame


def main() -> None:
    args = parse_args()
    src_path = Path(args.source)
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    frame = read_first_frame(str(src_path))
    disp = frame.copy()

    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("calib", disp)
            print(f"Clicked point {len(points)} at (u={x}, v={y})")

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.imshow("calib", disp)
    cv2.setMouseCallback("calib", on_mouse)

    print(
        "\nInstructions:\n"
        "  1) Click at least 4 points on the ROAD PLANE (e.g., lane corners).\n"
        "  2) When done, press 'q' in the window.\n"
        "     Recommended: choose a near-left, near-right, far-right, far-left corner\n"
        "     of one lane so you can assign real distances like:\n"
        "       (0,0), (3.5,0), (3.5,10), (0,10) in meters.\n"
    )

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(points) < 4:
        raise RuntimeError(f"Need at least 4 points, got {len(points)}.")

    print("\nYou clicked these image points (u,v):")
    for i, (u, v) in enumerate(points):
        print(f"  {i}: ({u:.1f}, {v:.1f})")

    # Ask for world coordinates in meters
    world_points = []
    print(
        "\nNow enter world coordinates (X,Y) in METERS for each point.\n"
        "Example if using 3.5 m lane width and 10 m patch along the road:\n"
        "  point 0 (near-left)  -> X=0,   Y=0\n"
        "  point 1 (near-right) -> X=3.5, Y=0\n"
        "  point 2 (far-right)  -> X=3.5, Y=10\n"
        "  point 3 (far-left)   -> X=0,   Y=10\n"
    )

    for i, (u, v) in enumerate(points):
        print(f"\nPoint {i} at image coords (u={u:.1f}, v={v:.1f})")
        X = float(input("  Enter world X (meters): "))
        Y = float(input("  Enter world Y (meters): "))
        world_points.append((X, Y))

    img_pts = np.array(points, dtype=np.float32)
    world_pts = np.array(world_points, dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, world_pts, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed to compute H.")

    out_h_path = Path(args.out_h)
    out_h_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_h_path, "w") as f:
        json.dump({"H": H.tolist()}, f, indent=2)

    out_p_path = Path(args.out_points)
    out_p_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p_path, "w") as f:
        json.dump(
            {
                "image_points": [[float(u), float(v)] for (u, v) in points],
                "world_points": [[float(X), float(Y)] for (X, Y) in world_points],
            },
            f,
            indent=2,
        )

    print("\nHomography H (image -> world, meters):")
    print(H)
    print(f"\nSaved homography to: {out_h_path.resolve()}")
    print(f"Saved correspondences to: {out_p_path.resolve()}")


if __name__ == "__main__":
    main()
