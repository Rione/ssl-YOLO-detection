from argparse import ArgumentParser
from pathlib import Path
import json

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "orange_ball_color_model.json"


def parse_args():
    parser = ArgumentParser(description="Detect orange balls with an OpenCV color model.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--source", default="0",
                        help="Camera index, image path, or video path. Default: 0")
    parser.add_argument("--min-area", type=int, default=300,
                        help="Minimum contour area to draw. Default: 300")
    parser.add_argument("--min-circularity", type=float, default=0.35,
                        help="Minimum circularity (0-1). Default: 0.35")
    parser.add_argument("--iou-merge", type=float, default=0.3,
                        help="IoU threshold for merging overlapping detections. Default: 0.3")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N frames. 0 = disable. Default: 10")
    return parser.parse_args()


def normalize_source(source):
    return int(source) if source.isdigit() else source


def load_color_model(path):
    model_path = Path(path)
    if not model_path.exists():
        print(f"Color model not found: {model_path}")
        print("Please run first: python3 train_orange_ball.py")
        return None
    model = json.loads(model_path.read_text(encoding="utf-8"))
    lower = np.array(model["lower"], dtype=np.uint8)
    upper = np.array(model["upper"], dtype=np.uint8)
    return lower, upper


def circle_iou(c1, c2):
    """Approximate IoU for two circles (cx, cy, r)."""
    cx1, cy1, r1 = c1
    cx2, cy2, r2 = c2
    dist = np.hypot(cx1 - cx2, cy1 - cy2)
    if dist >= r1 + r2:
        return 0.0
    if dist <= abs(r1 - r2):
        smaller_area = np.pi * min(r1, r2) ** 2
        union = np.pi * max(r1, r2) ** 2
        return smaller_area / union
    a = (r1**2 * np.arccos((dist**2 + r1**2 - r2**2) / (2 * dist * r1))
         + r2**2 * np.arccos((dist**2 + r2**2 - r1**2) / (2 * dist * r2))
         - 0.5 * np.sqrt((-dist + r1 + r2) * (dist + r1 - r2) * (dist - r1 + r2) * (dist + r1 + r2)))
    union = np.pi * (r1**2 + r2**2) - a
    return a / union if union > 0 else 0.0


def merge_overlapping_circles(circles, iou_threshold):
    """Merge circles whose IoU exceeds the threshold, keeping the largest."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[3], reverse=True)
    kept = []
    suppressed = [False] * len(circles)
    for i, ci in enumerate(circles):
        if suppressed[i]:
            continue
        kept.append(ci)
        for j in range(i + 1, len(circles)):
            if suppressed[j]:
                continue
            if circle_iou(ci[:3], circles[j][:3]) >= iou_threshold:
                suppressed[j] = True
    return kept


def detect_orange_balls(frame, lower, upper, min_area, min_circularity, iou_merge):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = np.ones((15, 15), dtype=np.uint8)
    kernel_open  = np.ones((5,  5),  dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)

    frame_height, frame_width = mask.shape[:2]
    edge_margin = 2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        touches_edge = (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= frame_width  - edge_margin
            or y + h >= frame_height - edge_margin
        )
        eff_circ = min_circularity * 0.5 if touches_edge else min_circularity

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < eff_circ:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), int(radius)

        candidates.append((cx, cy, radius, area, circularity))

    detections = merge_overlapping_circles(candidates, iou_merge)
    return detections, mask


def log_detections(frame_no, detections):
    """Print detection results to the terminal."""
    timestamp = f"[frame {frame_no:06d}]"
    if not detections:
        print(f"{timestamp} no detection")
        return
    print(f"{timestamp} detected: {len(detections)}")
    for i, (cx, cy, radius, area, circularity) in enumerate(detections, 1):
        print(
            f"  [{i}] center=({cx:4d}, {cy:4d})  radius={radius:4d}px"
            f"  area={area:8.0f}px2  circularity={circularity:.3f}"
        )


def draw_detections(frame, detections):
    color = (255, 100, 0)  # blue (BGR)

    for cx, cy, radius, area, circularity in detections:
        cv2.circle(frame, (cx, cy), radius, color, 2)
        label = f"orange ball  r={radius}  circ={circularity:.2f}"
        cv2.putText(
            frame,
            label,
            (cx - radius, max(20, cy - radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def main():
    args = parse_args()
    color_model = load_color_model(args.model)
    if color_model is None:
        return

    lower, upper = color_model
    cap = cv2.VideoCapture(normalize_source(args.source))

    if not cap.isOpened():
        print(f"Failed to open source: {args.source}")
        return

    print("Orange ball detection started. Press 'q' to quit, 'm' to toggle mask view.")
    show_mask = False
    mask_window_open = False
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, mask = detect_orange_balls(
            frame, lower, upper,
            args.min_area, args.min_circularity, args.iou_merge
        )
        draw_detections(frame, detections)

        if args.log_interval > 0 and frame_no % args.log_interval == 0:
            log_detections(frame_no, detections)

        frame_no += 1

        cv2.imshow("Orange Ball Detection", frame)
        if show_mask:
            cv2.imshow("Orange Ball Mask", mask)
            mask_window_open = True
        elif mask_window_open:
            cv2.destroyWindow("Orange Ball Mask")
            mask_window_open = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
