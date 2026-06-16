from argparse import ArgumentParser
from pathlib import Path
import json

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "orange_ball_color_model.json"


def parse_args():
    parser = ArgumentParser(description="Detect orange balls with an OpenCV color model.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to orange_ball_color_model.json.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, image path, or video path. Default: 0",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=300,
        help="Minimum contour area to draw. Default: 300",
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.6,
        help="Minimum circularity (0-1) to filter out non-round noise. Default: 0.6",
    )
    return parser.parse_args()


def normalize_source(source):
    return int(source) if source.isdigit() else source


def load_color_model(path):
    model_path = Path(path)
    if not model_path.exists():
        print(f"色モデルが見つかりません: {model_path}")
        print("先に次を実行してください: python3 train_orange_ball.py")
        return None

    model = json.loads(model_path.read_text(encoding="utf-8"))
    lower = np.array(model["lower"], dtype=np.uint8)
    upper = np.array(model["upper"], dtype=np.uint8)
    return lower, upper


def detect_orange_balls(frame, lower, upper, min_area, min_circularity):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    frame_height, frame_width = mask.shape[:2]
    edge_margin = 2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        # 画面端(上下左右)に接している輪郭は、ボールが見切れている可能性が
        # 高いため circularity 基準を緩める
        touches_edge = (
            x <= edge_margin
            or y <= edge_margin
            or x + width >= frame_width - edge_margin
            or y + height >= frame_height - edge_margin
        )
        effective_min_circularity = min_circularity * 0.5 if touches_edge else min_circularity

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < effective_min_circularity:
            continue

        detections.append((x, y, width, height, area, circularity))

    return detections, mask


def draw_detections(frame, detections):
    for x, y, width, height, area, circularity in detections:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 180, 255), 2)
        label = f"orange ball {area:.0f}"
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
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
        print(f"入力を開けませんでした: {args.source}")
        return

    print("オレンジボール検知を開始します。'q' キーで終了、'm' キーでマスク表示を切り替えます。")
    show_mask = False
    mask_window_open = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, mask = detect_orange_balls(frame, lower, upper, args.min_area, args.min_circularity)
        draw_detections(frame, detections)

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