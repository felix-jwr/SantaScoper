#!/usr/bin/env python3
"""
Script to visualise YOLO format bounding boxes on an image.
Reads bounding boxes from dummy.txt and draws them on dummy.jpg.
"""

import cv2
import numpy as np


def visualise_bboxes(image_path="./dataset/santascoper/images/val/0000001_02999_d_0000005.jpg",
                      label_path="./dataset/santascoper/labels/val/0000001_02999_d_0000005.txt",
                      output_path="dummy_visualised.jpg"):
    """
    Visualise YOLO format bounding boxes on an image.
    
    Args:
        image_path: Path to the input image
        label_path: Path to the label file (YOLO format)
        output_path: Path to save the output image with bounding boxes
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    img_h, img_w = img.shape[:2]
    print(f"Image size: {img_w}x{img_h}")
    
    # Define class names (update as needed)
    class_names = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor",
        10: "santa"
    }
    
    # Define colors for different classes (BGR format)
    colors = {
        0: (255, 0, 0),      # Blue
        1: (0, 255, 0),      # Green
        2: (0, 0, 255),      # Red
        3: (255, 255, 0),    # Cyan
        4: (255, 0, 255),    # Magenta
        5: (0, 255, 255),    # Yellow
        6: (128, 0, 128),    # Purple
        7: (0, 128, 128),    # Teal
        8: (128, 128, 0),    # Olive
        9: (192, 192, 192),  # Silver
        10: (255, 0, 127),   # Hot Pink (for Santa)
    }
    
    # Read bounding boxes from label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Label file {label_path} not found")
        return
    
    print(f"Found {len(lines)} bounding box(es)")
    
    # Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert from YOLO format (normalised) to pixel coordinates
        x_center_px = x_center * img_w
        y_center_px = y_center * img_h
        width_px = width * img_w
        height_px = height * img_h
        
        # Calculate top-left and bottom-right corners
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # Get color and class name
        color = colors.get(class_id, (0, 255, 0))  # Default to green
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(f"  Class: {class_name} (ID: {class_id})")
        print(f"  Bbox: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Size: {width_px:.1f}x{height_px:.1f} pixels")
    
    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"\nVisualised image saved to: {output_path}")


if __name__ == "__main__":
    visualise_bboxes()
