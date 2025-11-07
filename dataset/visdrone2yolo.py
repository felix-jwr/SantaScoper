import os
from pathlib import Path
import shutil

from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL, TQDM


def visdrone2yolo(dir, split, source_name=None):
    """Convert VisDrone annotations to YOLO format with images/{split} and labels/{split} structure."""
    from PIL import Image

    source_dir = dir / (source_name or f"VisDrone2019-DET-{split}")
    images_dir = dir / "images" / split
    labels_dir = dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images to new structure
    if (source_images_dir := source_dir / "images").exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in source_images_dir.glob("*.jpg"):
            img.rename(images_dir / img.name)

    for f in TQDM((source_dir / "annotations").glob("*.txt"), desc=f"Converting {split}"):
        img_size = Image.open(images_dir / f.with_suffix(".jpg").name).size
        dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
        lines = []

        with open(f, encoding="utf-8") as file:
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] != "0":  # Skip ignored regions
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1
                    # Convert to YOLO format
                    x_center, y_center = (x + w / 2) * dw, (y + h / 2) * dh
                    w_norm, h_norm = w * dw, h * dh
                    lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        (labels_dir / f.name).write_text("".join(lines), encoding="utf-8")


dir = Path("")
visdrone2yolo(dir, "val", "VisDrone2019-DET-val")  # convert VisDrone annotations to YOLO labels
shutil.rmtree(dir / "VisDrone2019-DET-val")  # cleanup original directory