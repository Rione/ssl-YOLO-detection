from argparse import ArgumentParser
from pathlib import Path
import json

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Orange Balls"
IMAGES_DIR = DATASET_DIR / "train" / "images"
LABELS_DIR = DATASET_DIR / "train" / "labels"
AUGMENTED_IMAGES_DIR = DATASET_DIR / "train_augmented" / "images"
AUGMENTED_LABELS_DIR = DATASET_DIR / "train_augmented" / "labels"
MODEL_PATH = ROOT / "orange_ball_color_model.json"


def parse_args():
    parser = ArgumentParser(description="Build an orange ball HSV color model from YOLO labels.")
    parser.add_argument(
        "--model-output",
        default=str(MODEL_PATH),
        help="Output JSON path. Default: orange_ball_color_model.json",
    )
    parser.add_argument(
        "--no-augmented",
        action="store_false",
        dest="include_augmented",
        help="Use only Orange Balls/train and ignore Orange Balls/train_augmented.",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=4.0,
        help="Lower HSV percentile. Smaller values make detection more tolerant. Default: 4",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=96.0,
        help="Upper HSV percentile. Larger values make detection more tolerant. Default: 96",
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
    image = cv2.imread(str(image_path))
    if image is not None:
        return image

    try:
        from PIL import Image, ImageOps

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass

        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def read_label_boxes(label_path, image_width, image_height):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        _, x_center, y_center, width, height = map(float, parts)
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def sample_ball_pixels(image, boxes):
    hsv_pixels = []

    for x1, y1, x2, y2 in boxes:
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        height, width = crop.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (max(1, int(width * 0.35)), max(1, int(height * 0.35)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        pixels = hsv[mask == 255]
        if len(pixels) > 0:
            hsv_pixels.append(pixels)

    if not hsv_pixels:
        return np.empty((0, 3), dtype=np.uint8)

    return np.concatenate(hsv_pixels, axis=0)


def dataset_parts(include_augmented):
    parts = [(IMAGES_DIR, LABELS_DIR)]
    if include_augmented and AUGMENTED_IMAGES_DIR.exists() and AUGMENTED_LABELS_DIR.exists():
        parts.append((AUGMENTED_IMAGES_DIR, AUGMENTED_LABELS_DIR))
    return parts


def build_color_model(parts, lower_percentile, upper_percentile):
    all_pixels = []
    used_images = 0
    skipped_images = 0
    source_dirs = []

    for images_dir, labels_dir in parts:
        source_dirs.append(str(labels_dir))
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image(label_path, images_dir)
            if image_path is None:
                skipped_images += 1
                continue

            image = read_image(image_path)
            if image is None:
                skipped_images += 1
                continue

            height, width = image.shape[:2]
            boxes = read_label_boxes(label_path, width, height)
            pixels = sample_ball_pixels(image, boxes)
            if len(pixels) == 0:
                skipped_images += 1
                continue

            all_pixels.append(pixels)
            used_images += 1

    if not all_pixels:
        raise RuntimeError("Orange Balls から色モデルを作れませんでした。JPG画像とラベルを確認してください。")

    pixels = np.concatenate(all_pixels, axis=0)
    lower = np.percentile(pixels, [lower_percentile], axis=0)[0]
    upper = np.percentile(pixels, [upper_percentile], axis=0)[0]

    lower = np.maximum(lower - np.array([5, 30, 30]), [0, 40, 40]).astype(int)
    upper = np.minimum(upper + np.array([5, 30, 30]), [179, 255, 255]).astype(int)

    return {
        "color_space": "HSV",
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "used_images": used_images,
        "skipped_images": skipped_images,
        "source_label_dirs": source_dirs,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
    }


def main():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise FileNotFoundError("Orange Balls/train/images と Orange Balls/train/labels が必要です。")

    args = parse_args()
    model = build_color_model(dataset_parts(args.include_augmented), args.lower_percentile, args.upper_percentile)
    model_path = Path(args.model_output)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(f"色モデルを保存しました: {model_path}")
    print(f"HSV lower={model['lower']} upper={model['upper']}")
    print(f"使用画像: {model['used_images']} / スキップ: {model['skipped_images']}")
    print(f"読み込みラベル: {', '.join(model['source_label_dirs'])}")


if __name__ == "__main__":
    main()
