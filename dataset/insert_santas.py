#!/usr/bin/env python3
"""
Script to randomly augment validation images with Santa.
- Adds Santa to validation images with probability 0.5
- Updates corresponding label files with Santa bounding boxes
- Santa class ID: 10
"""

import cv2
import numpy as np
from pathlib import Path
from augment_santa import SantaAugmentation


def insert_santas(
    images_dir="./dataset/santascoper/images/val",
    labels_dir="./dataset/santascoper/labels/val",
    santa_images_dir="./dataset/santa_images",
    probability=0.5,
    santa_class_id=10,
):
    """
    Augment validation images with Santa and update label files.
    
    Args:
        images_dir: Directory containing validation images
        labels_dir: Directory containing validation label files
        santa_images_dir: Directory containing Santa PNG images
        probability: Probability of adding Santa to each image (0.0-1.0)
        santa_class_id: Class ID for Santa in YOLO format
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Initialize Santa augmentation
    santa_aug = SantaAugmentation(
        santa_images_dir=santa_images_dir,
        max_santa_ratio=0.4,
        p=probability
    )
    
    # Get all image files
    image_extensions = ['.jpg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f'*{ext}'))
    
    total_images = len(image_files)
    santa_added_count = 0
    
    print(f"Found {total_images} images in {images_dir}")
    print(f"Santa augmentation probability: {probability}")
    print(f"Santa class ID: {santa_class_id}")
    print(f"Processing images...\n")
    
    for idx, image_path in enumerate(image_files, 1):
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path.name}")
            continue
        
        # Apply Santa augmentation
        aug_img, santa_bbox = santa_aug(img, bboxes=None, return_bbox=True)
        
        # Check if Santa was added
        if santa_bbox is not None:
            # Save augmented image
            print(str(image_path))
            cv2.imwrite(str(image_path), aug_img)
            
            # Update label file
            label_path = labels_path / f"{image_path.stem}.txt"
            
            # Convert bbox from pixels to YOLO format
            img_h, img_w = img.shape[:2]
            x_min, y_min, w, h = santa_bbox
            
            # Calculate YOLO format (normalized center coordinates)
            x_center = (x_min + w / 2) / img_w
            y_center = (y_min + h / 2) / img_h
            width_norm = w / img_w
            height_norm = h / img_h

            # Append to label file
            with open(label_path, 'a') as f:
                f.write(f"{santa_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            
            santa_added_count += 1
            print(f"[{idx}/{total_images}] Added Santa to {image_path.name}")
        else:
            print(f"[{idx}/{total_images}] No Santa added to {image_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Santa added to: {santa_added_count} images ({santa_added_count/total_images*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    insert_santas()
