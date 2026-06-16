"""
位置・形状のみを変化させる水増しスクリプト。

色相・彩度・明度のジッターやノイズ・ぼかしは一切加えない。
HSV色モデル(train_orange_ball.py)の学習データとして使っても、
オレンジ色の分布を歪めにくい。

変化させる要素:
- 回転 (rotation)
- スケール (scale)
- 平行移動 (translation)
- 左右/上下反転 (flip)
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Orange Balls"
SOURCE_IMAGES_DIR = DATASET_DIR / "train" / "images"
SOURCE_LABELS_DIR = DATASET_DIR / "train" / "labels"
OUTPUT_DIR = DATASET_DIR / "train_augmented_geo"


def parse_args():
    parser = ArgumentParser(description="Create geometry-only augmented YOLO images for orange ball training.")
    parser.add_argument(
        "--source-images",
        default=str(SOURCE_IMAGES_DIR),
        help="Input image directory. Default: Orange Balls/train/images",
    )
    parser.add_argument(
        "--source-labels",
        default=str(SOURCE_LABELS_DIR),
        help="Input YOLO label directory. Default: Orange Balls/train/labels",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output split directory. Default: Orange Balls/train_augmented_geo",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=3,
        help="Number of augmented images to create per source image. Default: 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for repeatable augmentation. Default: 7",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit source images for quick tests. 0 means no limit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing augmented images and labels.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write augmented files into the source train/images and train/labels folders.",
    )
    parser.add_argument(
        "--max-rotation",
        type=float,
        default=10.0,
        help="Max rotation angle in degrees (each direction). Default: 10",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.85,
        help="Minimum scale factor. Default: 0.85",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.15,
        help="Maximum scale factor. Default: 1.15",
    )
    parser.add_argument(
        "--max-translation",
        type=float,
        default=0.08,
        help="Max translation as a fraction of image size. Default: 0.08",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="Probability of horizontal flip. Default: 0.5",
    )
    parser.add_argument(
        "--vflip-prob",
        type=float,
        default=0.12,
        help="Probability of vertical flip. Default: 0.12",
    )
    return parser.parse_args()


def find_image(label_path, images_dir):
    for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        image_path = images_dir / f"{label_path.stem}{suffix}"
        if image_path.exists():
            return image_path

    matches = sorted(images_dir.glob(f"{label_path.stem}.*"))
    if matches:
        return matches[0]
    return None


def read_image(image_path):
    """JPG/PNG専用。HEICは事前にconvert_heic_to_jpg.pyで変換しておくこと。"""
    return cv2.imread(str(image_path))


def read_yolo_labels(label_path):
    labels = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = parts[0]
        try:
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            continue

        if width <= 0 or height <= 0:
            continue

        labels.append((class_id, x_center, y_center, width, height))

    return labels


def labels_to_boxes(labels, image_width, image_height):
    boxes = []

    for class_id, x_center, y_center, width, height in labels:
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height
        boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def boxes_to_labels(boxes, image_width, image_height):
    labels = []

    for class_id, x1, y1, x2, y2 in boxes:
        x1 = max(0.0, min(float(image_width - 1), x1))
        y1 = max(0.0, min(float(image_height - 1), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height

        if box_width <= 0.001 or box_height <= 0.001:
            continue

        labels.append((class_id, x_center, y_center, box_width, box_height))

    return labels


def format_yolo_labels(labels):
    return "\n".join(
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for class_id, x_center, y_center, width, height in labels
    ) + "\n"


def transform_boxes_affine(boxes, matrix, image_width, image_height):
    transformed = []

    for class_id, x1, y1, x2, y2 in boxes:
        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        )
        moved = corners @ matrix.T
        nx1, ny1 = moved[:, 0].min(), moved[:, 1].min()
        nx2, ny2 = moved[:, 0].max(), moved[:, 1].max()

        nx1 = max(0.0, min(float(image_width - 1), nx1))
        ny1 = max(0.0, min(float(image_height - 1), ny1))
        nx2 = max(0.0, min(float(image_width), nx2))
        ny2 = max(0.0, min(float(image_height), ny2))

        old_area = max(1.0, (x2 - x1) * (y2 - y1))
        new_area = max(0.0, (nx2 - nx1) * (ny2 - ny1))
        if new_area / old_area >= 0.25:
            transformed.append((class_id, nx1, ny1, nx2, ny2))

    return transformed


def apply_affine(image, boxes, rng, args):
    height, width = image.shape[:2]
    angle = rng.uniform(-args.max_rotation, args.max_rotation)
    scale = rng.uniform(args.scale_min, args.scale_max)
    tx = rng.uniform(-args.max_translation, args.max_translation) * width
    ty = rng.uniform(-args.max_translation, args.max_translation) * height

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += [tx, ty]

    augmented = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return augmented, transform_boxes_affine(boxes, matrix, width, height)


def apply_flips(image, boxes, rng, args):
    height, width = image.shape[:2]

    if rng.random() < args.hflip_prob:
        image = cv2.flip(image, 1)
        boxes = [(class_id, width - x2, y1, width - x1, y2) for class_id, x1, y1, x2, y2 in boxes]

    if rng.random() < args.vflip_prob:
        image = cv2.flip(image, 0)
        boxes = [(class_id, x1, height - y2, x2, height - y1) for class_id, x1, y1, x2, y2 in boxes]

    return image, boxes


def augment_image(image, labels, rng, args):
    height, width = image.shape[:2]
    boxes = labels_to_boxes(labels, width, height)

    image, boxes = apply_affine(image, boxes, rng, args)
    image, boxes = apply_flips(image, boxes, rng, args)
    labels = boxes_to_labels(boxes, width, height)
    if not labels:
        return None, []

    return image, labels


def main():
    args = parse_args()
    source_images = Path(args.source_images)
    source_labels = Path(args.source_labels)

    if args.in_place:
        output_images = source_images
        output_labels = source_labels
    else:
        output_root = Path(args.output_dir)
        output_images = output_root / "images"
        output_labels = output_root / "labels"

    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError("入力の images と labels フォルダが見つかりません。")

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    label_paths = sorted(source_labels.glob("*.txt"))
    if args.max_images > 0:
        label_paths = label_paths[: args.max_images]

    created = 0
    skipped = 0

    for label_path in label_paths:
        if "_aug" in label_path.stem:
            skipped += 1
            continue

        image_path = find_image(label_path, source_images)
        if image_path is None:
            skipped += 1
            continue

        image = read_image(image_path)
        labels = read_yolo_labels(label_path)
        if image is None or not labels:
            skipped += 1
            continue

        for copy_index in range(1, args.copies + 1):
            output_stem = f"{label_path.stem}_aug{copy_index:02d}"
            output_image_path = output_images / f"{output_stem}.jpg"
            output_label_path = output_labels / f"{output_stem}.txt"

            if not args.force and output_image_path.exists() and output_label_path.exists():
                continue

            augmented, augmented_labels = augment_image(image, labels, rng, args)
            if augmented is None:
                skipped += 1
                continue

            cv2.imwrite(str(output_image_path), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
            output_label_path.write_text(format_yolo_labels(augmented_labels), encoding="utf-8")
            created += 1

    print(f"水増し画像を作成しました: {created} 件")
    print(f"スキップ: {skipped} 件")
    print(f"画像: {output_images}")
    print(f"ラベル: {output_labels}")


if __name__ == "__main__":
    main()