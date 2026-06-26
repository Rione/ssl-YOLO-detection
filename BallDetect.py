"""
オレンジボール検出スクリプト。

train_orange_ball.py で生成された last.pt (または best.pt) を使って
カメラ・動画・静止画からオレンジボールを検出する。

使い方:
    python BallDetect.py                          # カメラ0 + last.pt
    python BallDetect.py --model best.pt          # best.pt を使う
    python BallDetect.py --source video.mp4       # 動画ファイル
    python BallDetect.py --source image.jpg       # 静止画
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
from ultralytics import YOLO


# ─── パス設定 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# スクリプトと同じディレクトリの last.pt を優先、なければ best.pt にフォールバック
def _default_model() -> Path:
    for name in ("last.pt", "best.pt"):
        p = ROOT / name
        if p.exists():
            return p
    return ROOT / "last.pt"   # 存在しない場合はエラーメッセージで案内


# ─── 引数 ────────────────────────────────────────────────────────────────────
def parse_args():
    parser = ArgumentParser(description="Detect orange balls with a YOLO .pt model.")
    parser.add_argument(
        "--model",
        default=str(_default_model()),
        help="Path to YOLO .pt model. Default: last.pt (same directory as this script)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (0,1,...), image path, or video path. Default: 0",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N frames (0 = no log). Default: 10",
    )
    return parser.parse_args()


# ─── ユーティリティ ──────────────────────────────────────────────────────────

def normalize_source(source: str):
    """数字文字列はカメラインデックス（int）に変換する。"""
    return int(source) if source.isdigit() else source


def log_detections(frame_no: int, boxes) -> None:
    ts = f"[frame {frame_no:06d}]"
    if len(boxes) == 0:
        print(f"{ts} no detection")
        return
    print(f"{ts} detected: {len(boxes)}")
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        print(f"  [{i}] bbox=[{x1}, {y1}, {x2}, {y2}]  conf={conf:.3f}  class={cls}")


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] モデルが見つかりません: {model_path}")
        print("  先に train_orange_ball.py を実行して last.pt を生成してください。")
        return

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    source = normalize_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] ソースを開けませんでした: {args.source}")
        return

    print("Orange ball detection started. Press 'q' to quit.")
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)
        boxes   = results[0].boxes
        annotated_frame = results[0].plot()

        if args.log_interval > 0 and frame_no % args.log_interval == 0:
            log_detections(frame_no, boxes)

        frame_no += 1

        cv2.imshow("Orange Ball Detection (YOLO)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()