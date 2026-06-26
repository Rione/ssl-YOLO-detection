"""
オレンジボール検出モデルの学習スクリプト。

- geometry augmentation（回転・スケール・平行移動・反転のみ）を内包
- 色調整は一切行わない（HSV分布を歪めないため）
- valid フォルダが存在しない場合、train から自動分割して用意する
- 学習完了後、last.pt / best.pt をこのスクリプトと同じディレクトリに出力する

使い方:
    python train_orange_ball.py

オプション:
    --epochs        学習エポック数 (デフォルト: 50)
    --imgsz         入力画像サイズ (デフォルト: 640)
    --device        デバイス: cpu / mps / 0  (デフォルト: cpu)
    --copies        1枚あたり水増し枚数 (デフォルト: 3)
    --val-split     valid が無い場合に train から切り出す割合 (デフォルト: 0.15)
    --no-augment    geometry augment をスキップする
"""

import random
import shutil
import textwrap
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ─── パス設定 ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Orange Balls"
DATA_YAML   = DATASET_DIR / "data.yaml"


# ─── 引数 ────────────────────────────────────────────────────────────────────
def parse_args():
    parser = ArgumentParser(description="Train YOLO orange ball detector.")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--imgsz",           type=int,   default=640)
    parser.add_argument("--device",          type=str,   default="cpu",
                        help="cpu / mps / 0 (CUDA)")
    parser.add_argument("--copies",          type=int,   default=3,
                        help="Geometry augmentation copies per image.")
    parser.add_argument("--val-split",       type=float, default=0.15,
                        help="Fraction of train to use as val when valid/ is missing.")
    parser.add_argument("--no-augment",      action="store_true",
                        help="Skip geometry augmentation.")
    parser.add_argument("--max-rotation",    type=float, default=10.0)
    parser.add_argument("--scale-min",       type=float, default=0.85)
    parser.add_argument("--scale-max",       type=float, default=1.15)
    parser.add_argument("--max-translation", type=float, default=0.08)
    parser.add_argument("--hflip-prob",      type=float, default=0.5)
    parser.add_argument("--vflip-prob",      type=float, default=0.12)
    parser.add_argument("--seed",            type=int,   default=7)
    return parser.parse_args()


# ─── Geometry Augmentation ───────────────────────────────────────────────────

def _find_image(label_path: Path, images_dir: Path) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = images_dir / f"{label_path.stem}{suffix}"
        if p.exists():
            return p
    matches = sorted(images_dir.glob(f"{label_path.stem}.*"))
    return matches[0] if matches else None


def _read_yolo_labels(label_path: Path):
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = parts[0]
        try:
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        if w > 0 and h > 0:
            labels.append((class_id, x, y, w, h))
    return labels


def _labels_to_boxes(labels, W, H):
    boxes = []
    for cid, xc, yc, bw, bh in labels:
        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W
        y2 = (yc + bh / 2) * H
        boxes.append((cid, x1, y1, x2, y2))
    return boxes


def _boxes_to_labels(boxes, W, H):
    labels = []
    for cid, x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(float(W - 1), x1))
        y1 = max(0.0, min(float(H - 1), y1))
        x2 = max(0.0, min(float(W), x2))
        y2 = max(0.0, min(float(H), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        if bw > 0.001 and bh > 0.001:
            labels.append((cid, xc, yc, bw, bh))
    return labels


def _transform_boxes(boxes, matrix, W, H):
    result = []
    for cid, x1, y1, x2, y2 in boxes:
        corners = np.array(
            [[x1, y1, 1.], [x2, y1, 1.], [x2, y2, 1.], [x1, y2, 1.]],
            dtype=np.float32,
        )
        moved = corners @ matrix.T
        nx1, ny1 = moved[:, 0].min(), moved[:, 1].min()
        nx2, ny2 = moved[:, 0].max(), moved[:, 1].max()
        nx1 = max(0., min(float(W - 1), nx1))
        ny1 = max(0., min(float(H - 1), ny1))
        nx2 = max(0., min(float(W), nx2))
        ny2 = max(0., min(float(H), ny2))
        old_area = max(1., (x2 - x1) * (y2 - y1))
        new_area = max(0., (nx2 - nx1) * (ny2 - ny1))
        if new_area / old_area >= 0.25:
            result.append((cid, nx1, ny1, nx2, ny2))
    return result


def _augment_one(image, labels, rng, args):
    H, W = image.shape[:2]
    boxes = _labels_to_boxes(labels, W, H)

    angle = rng.uniform(-args.max_rotation, args.max_rotation)
    scale = rng.uniform(args.scale_min, args.scale_max)
    tx = rng.uniform(-args.max_translation, args.max_translation) * W
    ty = rng.uniform(-args.max_translation, args.max_translation) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[:, 2] += [tx, ty]
    image = cv2.warpAffine(image, M, (W, H),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    boxes = _transform_boxes(boxes, M, W, H)

    if rng.random() < args.hflip_prob:
        image = cv2.flip(image, 1)
        boxes = [(cid, W - x2, y1, W - x1, y2) for cid, x1, y1, x2, y2 in boxes]
    if rng.random() < args.vflip_prob:
        image = cv2.flip(image, 0)
        boxes = [(cid, x1, H - y2, x2, H - y1) for cid, x1, y1, x2, y2 in boxes]

    labels_out = _boxes_to_labels(boxes, W, H)
    return (image, labels_out) if labels_out else (None, [])


def _format_labels(labels):
    return "\n".join(
        f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        for cid, xc, yc, bw, bh in labels
    ) + "\n"


def run_geometry_augmentation(train_images: Path, train_labels: Path,
                               out_images: Path, out_labels: Path,
                               args) -> int:
    """geometry augmentation を実行し、生成枚数を返す。"""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    created = 0

    for label_path in sorted(train_labels.glob("*.txt")):
        if "_aug" in label_path.stem:
            continue
        img_path = _find_image(label_path, train_images)
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        labels = _read_yolo_labels(label_path)
        if image is None or not labels:
            continue

        for i in range(1, args.copies + 1):
            stem = f"{label_path.stem}_aug{i:02d}"
            aug_img, aug_labels = _augment_one(image, labels, rng, args)
            if aug_img is None:
                continue
            cv2.imwrite(str(out_images / f"{stem}.jpg"), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_labels / f"{stem}.txt").write_text(
                _format_labels(aug_labels), encoding="utf-8"
            )
            created += 1

    return created


# ─── valid フォルダの確認・自動分割 ─────────────────────────────────────────

def ensure_valid_dir(train_images: Path, train_labels: Path,
                     val_split: float, seed: int) -> Path:
    """
    valid/images と valid/labels が存在しない場合、
    train から val_split 割合のファイルをコピーして作成する。
    既に valid/images に1枚以上あればそのまま返す。
    """
    val_images = DATASET_DIR / "valid" / "images"
    val_labels = DATASET_DIR / "valid" / "labels"

    # すでに valid/images が存在して中身があれば何もしない
    if val_images.exists() and any(val_images.iterdir()):
        print(f"      valid/ が存在します ({sum(1 for _ in val_images.iterdir())} 枚) → そのまま使用")
        return DATASET_DIR / "valid"

    print(f"      valid/ が見つかりません。train から {val_split:.0%} を分割して作成します...")
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(train_labels.glob("*.txt"))
    # aug 画像は val に入れない（元画像のみで分割）
    original_labels = [p for p in label_paths if "_aug" not in p.stem]

    rng_split = random.Random(seed)
    rng_split.shuffle(original_labels)
    n_val = max(1, int(len(original_labels) * val_split))
    val_label_paths = original_labels[:n_val]

    copied = 0
    for lp in val_label_paths:
        img_path = _find_image(lp, train_images)
        if img_path is None:
            continue
        shutil.copy2(img_path, val_images / img_path.name)
        shutil.copy2(lp,       val_labels / lp.name)
        copied += 1

    print(f"      valid/ に {copied} 枚をコピーしました → {DATASET_DIR / 'valid'}")
    return DATASET_DIR / "valid"


# ─── data.yaml 生成 ──────────────────────────────────────────────────────────

def build_data_yaml(train_dir: Path, val_dir: Path,
                    nc: int, names: list) -> str:
    names_str = "[" + ", ".join(str(n) for n in names) + "]"
    return textwrap.dedent(f"""\
        train: {train_dir / 'images'}
        val:   {val_dir / 'images'}
        nc: {nc}
        names: {names_str}
    """)


# ─── 最新の runs フォルダから weights を探す ────────────────────────────────

def find_latest_weights(runs_detect: Path) -> Path | None:
    """
    runs/detect/ 以下の orange_ball* フォルダのうち最新のものの
    weights/ ディレクトリを返す。
    """
    candidates = sorted(runs_detect.glob("orange_ball*/weights"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} が見つかりません。")
        return

    import yaml  # ultralytics に同梱されている PyYAML
    with open(DATA_YAML, encoding="utf-8") as f:
        base_yaml = yaml.safe_load(f)
    nc    = int(base_yaml.get("nc", 1))
    names = base_yaml.get("names", ["orange_ball"])

    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"

    # ── [1/3] Geometry augmentation ──
    if not args.no_augment:
        aug_root   = DATASET_DIR / "train_augmented_geo"
        aug_images = aug_root / "images"
        aug_labels = aug_root / "labels"

        print("[1/3] Geometry augmentation を実行中...")
        n = run_geometry_augmentation(
            train_images, train_labels, aug_images, aug_labels, args
        )
        print(f"      水増し画像: {n} 枚 → {aug_root}")

        # 元画像 + aug画像 を _train_merged/ に統合
        tmp_root   = DATASET_DIR / "_train_merged"
        tmp_images = tmp_root / "images"
        tmp_labels = tmp_root / "labels"
        tmp_images.mkdir(parents=True, exist_ok=True)
        tmp_labels.mkdir(parents=True, exist_ok=True)

        for src in list(train_images.glob("*")) + list(aug_images.glob("*")):
            dst = tmp_images / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        for src in list(train_labels.glob("*.txt")) + list(aug_labels.glob("*.txt")):
            dst = tmp_labels / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

        effective_train = tmp_root
    else:
        print("[1/3] Geometry augmentation をスキップします。")
        effective_train = DATASET_DIR / "train"

    # ── [2/3] valid の確認・自動分割 + data.yaml 生成 ──
    print("[2/3] valid データを確認中...")
    val_dir = ensure_valid_dir(train_images, train_labels, args.val_split, args.seed)

    tmp_yaml_path = ROOT / "_tmp_data.yaml"
    tmp_yaml_path.write_text(
        build_data_yaml(effective_train, val_dir, nc, names),
        encoding="utf-8",
    )
    print(f"      _tmp_data.yaml を生成しました")

    # ── [3/3] YOLO 学習 ──
    print(f"[3/3] 学習開始: epochs={args.epochs}, imgsz={args.imgsz}, device={args.device}")
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(tmp_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name="orange_ball",
    )

    # ── last.pt / best.pt をスクリプトと同じディレクトリにコピー ──
    #    runs/detect/ 以下の orange_ball* から最新の weights/ を探す
    runs_detect = ROOT / "runs" / "detect"
    weights_dir = find_latest_weights(runs_detect)

    if weights_dir is None:
        print("[WARN] weights フォルダが見つかりませんでした。runs/ を確認してください。")
    else:
        print(f"      weights: {weights_dir}")
        for weight_name in ("last.pt", "best.pt"):
            src = weights_dir / weight_name
            dst = ROOT / weight_name
            if src.exists():
                shutil.copy2(src, dst)
                print(f"✓ {weight_name} → {dst}")
            else:
                print(f"  {weight_name} は見つかりませんでした。")

    # 一時 yaml を削除
    tmp_yaml_path.unlink(missing_ok=True)

    print("\n=== 学習完了 ===")
    print(f"  last.pt : {ROOT / 'last.pt'}")
    print(f"  best.pt : {ROOT / 'best.pt'}")
    print("BallDetect.py でそのまま使えます。")


if __name__ == "__main__":
    main()