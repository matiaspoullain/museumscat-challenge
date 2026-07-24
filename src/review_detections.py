import csv
import os

import cv2

from src.detect_labels import _deskew, find_search_roi

IMAGES_DIR = os.path.join("data", "raw", "images")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")
OUT_DIR = os.path.join("tables_figures", "eda", "detection_review")


def sample_train_files(n=20):
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["image_file"] for row in rows[:n]]


def draw_roi(image_path, angle, box):
    img = cv2.imread(image_path)
    img, _ = _deskew(img)
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 8)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for image_file in sample_train_files():
        image_path = os.path.join(IMAGES_DIR, image_file)
        angle, box = find_search_roi(image_path)
        annotated = draw_roi(image_path, angle, box)
        out_path = os.path.join(OUT_DIR, image_file)
        h, w = annotated.shape[:2]
        annotated = cv2.resize(annotated, (w // 4, h // 4))
        cv2.imwrite(out_path, annotated)
        print(f"{image_file}: angle={angle:.2f} box={box}")


if __name__ == "__main__":
    main()
