import cv2
import random
import numpy as np
from pathlib import Path


class SantaAugmentation:
    """
    Augmentation that inserts Santa images with random probability.
    Handles transparency, positioning, scaling, and ground truth bounding boxes.
    """
    
    def __init__(self, santa_images_dir="./dataset/santa_images", max_santa_ratio=0.4, p=0.5):
        """
        Initialise the Santa augmentation transformer.
        
        Args:
            santa_images_dir: Directory containing Santa PNG images
            max_santa_ratio: Maximum size of Santa relative to base image (0.0-1.0)
            p: Probability of applying the augmentation (0.0-1.0)
        """
        self.santa_images_dir = Path(santa_images_dir)
        self.max_santa_ratio = max_santa_ratio
        self.p = p
        self.santa_bbox = None
        
        self.santa_images = self._load_santa_images()
        
        if not self.santa_images:
            print(f"Warning: No Santa images found in {self.santa_images_dir}")
    
    def _load_santa_images(self):
        """Load all Santa images from the specified directory."""
        santa_images = []
        if self.santa_images_dir.exists():
            for img_path in self.santa_images_dir.glob("*.png"):
                santa_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if santa_img is not None:
                    santa_images.append(santa_img)
        return santa_images

    def __call__(self, base_image, bboxes=None, return_bbox=True):
        """
        Apply Santa augmentation to image and optionally append to bounding boxes.
        
        Args:
            base_image: Input image (numpy array)
            bboxes: Optional list of bounding boxes in YOLO format 
                    [class_id, x_centre, y_centre, width, height]
            return_bbox: If True, return bounding box information; if False, return only image
        
        Returns:
            If return_bbox is False: augmented_image
            If return_bbox is True and bboxes is None: (augmented_image, santa_bbox_xywh_pixels or None)
            If return_bbox is True and bboxes is provided: (augmented_image, updated_bboxes_list)
        """
        if not self._should_apply():
            return self._return_unchanged(base_image, bboxes, return_bbox)
        
        if not self.santa_images:
            return self._return_unchanged(base_image, bboxes, return_bbox)
        
        aug_image, bbox = self._insert_santa(base_image)
        self.santa_bbox = bbox
        
        if not return_bbox:
            return aug_image
        elif bboxes is None:
            return aug_image, bbox
        else:
            updated_bboxes = self._append_santa_to_bboxes(bboxes, base_image.shape)
            return aug_image, updated_bboxes
    
    def _should_apply(self):
        """Determine whether augmentation should be applied based on probability."""
        return random.random() <= self.p
    
    def _return_unchanged(self, base_image, bboxes, return_bbox):
        """Return unchanged image and bounding boxes."""
        self.santa_bbox = None
        if not return_bbox:
            return base_image
        elif bboxes is None:
            return base_image, None
        else:
            return base_image, bboxes
    
    def _append_santa_to_bboxes(self, bboxes, image_shape):
        """
        Append Santa bounding box to existing bounding boxes in YOLO format.
        
        Args:
            bboxes: List of bounding boxes in YOLO format 
                    [class_id, x_centre, y_centre, width, height]
            image_shape: Tuple of (height, width) or (height, width, channels)
        
        Returns:
            Updated list of bounding boxes with Santa bounding box appended
        """
        if self.santa_bbox is None:
            return bboxes
        
        img_h, img_w = image_shape[:2]
        x_min, y_min, w, h = self.santa_bbox
        
        # Convert to normalised YOLO format (centre-based)
        x_centre = (x_min + w / 2) / img_w
        y_centre = (y_min + h / 2) / img_h
        width_norm = w / img_w
        height_norm = h / img_h
        
        santa_bbox = [80, x_centre, y_centre, width_norm, height_norm]
        return bboxes + [santa_bbox] if len(bboxes) > 0 else [santa_bbox]
    
    def _insert_santa(self, base_image):
        """
        Insert a random Santa image at a random location on the base image.
        
        Args:
            base_image: Base image to augment
            
        Returns:
            Tuple of (augmented_image, bbox_xywh_pixels)
        """
        base_h, base_w = base_image.shape[:2]
        santa_img = random.choice(self.santa_images).copy()
        
        santa_img = self._scale_santa(santa_img, base_w, base_h)
        new_h, new_w = santa_img.shape[:2]
        
        x_min, y_min = self._get_random_position(base_w, base_h, new_w, new_h)
        y_max = y_min + new_h
        x_max = x_min + new_w
        
        aug_image = self._blend_santa(base_image.copy(), santa_img, x_min, y_min, x_max, y_max)
        
        return aug_image, [x_min, y_min, new_w, new_h]
    
    def _scale_santa(self, santa_img, base_w, base_h):
        """Scale Santa image to fit within max_santa_ratio of the base image."""
        santa_h, santa_w = santa_img.shape[:2]
        santa_ratio = np.random.uniform(0.1, self.max_santa_ratio)
        max_santa_w = int(base_w * santa_ratio)
        max_santa_h = int(base_h * santa_ratio)
        
        scale_w = max_santa_w / santa_w
        scale_h = max_santa_h / santa_h
        scale = min(scale_w, scale_h)
        
        new_w = int(santa_w * scale)
        new_h = int(santa_h * scale)
        
        return cv2.resize(santa_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _get_random_position(self, base_w, base_h, santa_w, santa_h):
        """Generate random position for Santa within image bounds."""
        y_min = np.random.randint(0, max(1, base_h - santa_h))
        x_min = np.random.randint(0, max(1, base_w - santa_w))
        return x_min, y_min
    
    def _blend_santa(self, base_image, santa_img, x_min, y_min, x_max, y_max):
        """Blend Santa image onto base image with alpha transparency if available."""
        if santa_img.shape[2] == 4:
            alpha = santa_img[:, :, 3:4] / 255.0
            rgb = santa_img[:, :, :3]
            base_image[y_min:y_max, x_min:x_max] = (
                rgb * alpha + base_image[y_min:y_max, x_min:x_max] * (1 - alpha)
            ).astype(np.uint8)
        else:
            base_image[y_min:y_max, x_min:x_max] = santa_img[:, :, :3]
        
        return base_image