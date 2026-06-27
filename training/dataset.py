# -*- coding: utf-8 -*-
"""
Dataset for training satellite image segmentation model.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SatelliteDataset(Dataset):
    """
    Dataset for satellite images with segmentation masks.

    Directory structure:
        data/
            train/
                images/
                    image_001.png
                    image_002.png
                masks/
                    image_001.png  (mask with values 0-3)
                    image_002.png
            val/
                images/
                masks/

    Masks should contain pixel values:
        0 - Other (background)
        1 - Urban areas
        2 - Water
        3 - Vegetation
    """

    def __init__(self, images_dir, masks_dir, transform=None, image_size=256,
                 crop=False, augment=False, balanced_crop=False,
                 rare_classes=(1, 2), balanced_attempts=10,
                 balanced_min_fraction=0.05, center_crop=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size

        # Spatial strategy:
        #   crop=False -> resize whole image to image_size
        #   crop=True  -> take a native-resolution crop of image_size
        #       balanced_crop=True -> retry crops until a rare class is present
        #       center_crop=True   -> deterministic center crop (use on validation)
        self.crop = crop
        self.balanced_crop = balanced_crop and crop
        self.center_crop = center_crop and crop
        self.rare_classes = np.asarray(rare_classes)
        self.balanced_attempts = balanced_attempts
        self.balanced_min_fraction = balanced_min_fraction

        # Legacy: a full albumentations pipeline may still be passed in. When set,
        # it takes priority and the flag-based path below is skipped.
        self.transform = transform

        # Flag-based pipeline: spatial step is handled manually, post-transforms
        # (augmentation + normalize + tensor) run on the already-sized patch.
        self._resize = A.Resize(image_size, image_size)
        self._pad = A.PadIfNeeded(min_height=image_size, min_width=image_size)
        self._post = _post_transforms(augment)

        self.images = sorted([f for f in os.listdir(images_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        print(f"Found {len(self.images)} images in {images_dir}")
    
    def __len__(self):
        return len(self.images)

    def _choose_crop(self, mask):
        """Return (top, left) for the crop window.

        - center_crop: deterministic center (stable validation metrics).
        - balanced_crop: sample several windows and keep the first that contains
          at least `balanced_min_fraction` of rare-class pixels; if none qualifies
          within `balanced_attempts`, keep the window with the most rare pixels.
        - otherwise: a single uniform random window.
        """
        h, w = mask.shape[:2]
        size = self.image_size
        max_top = max(h - size, 0)
        max_left = max(w - size, 0)

        if self.center_crop:
            return max_top // 2, max_left // 2

        if not self.balanced_crop:
            return random.randint(0, max_top), random.randint(0, max_left)

        best_coords = (0, 0)
        best_frac = -1.0
        for _ in range(self.balanced_attempts):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            sub = mask[top:top + size, left:left + size]
            frac = float(np.isin(sub, self.rare_classes).mean())
            if frac >= self.balanced_min_fraction:
                return top, left
            if frac > best_frac:
                best_frac = frac
                best_coords = (top, left)
        return best_coords

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        mask_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            potential_path = os.path.join(self.masks_dir, mask_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

        if mask_path is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

        # Legacy path: a prebuilt albumentations Compose was provided.
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask'].long()

        if self.crop:
            if image.shape[0] < self.image_size or image.shape[1] < self.image_size:
                padded = self._pad(image=image, mask=mask)
                image, mask = padded['image'], padded['mask']
            top, left = self._choose_crop(mask)
            s = self.image_size
            image = image[top:top + s, left:left + s]
            mask = mask[top:top + s, left:left + s]
        else:
            resized = self._resize(image=image, mask=mask)
            image, mask = resized['image'], resized['mask']

        out = self._post(image=image, mask=mask)
        return out['image'], out['mask'].long()


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def _post_transforms(augment):
    """Transforms applied AFTER the spatial step (crop/resize).

    The spatial step is handled inside the dataset so we can do balanced and
    deterministic cropping; here we only add augmentation (train only),
    normalization and tensor conversion.
    """
    augs = []
    if augment:
        augs = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.ColorJitter(p=1),
            ], p=0.3),
            A.GaussNoise(p=0.2),
        ]
    return A.Compose(augs + [
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def _spatial_transform(image_size, crop):
    """Legacy helper kept for backward compatibility (no longer used by the
    flag-based dataset path).

    - crop=False: resize the whole image (objects shrink; small classes like
      buildings can disappear on large satellite tiles).
    - crop=True: take a random native-resolution crop, so small structures keep
      their real scale. This must match tiled inference at prediction time.
    """
    if crop:
        return [
            A.PadIfNeeded(min_height=image_size, min_width=image_size),
            A.RandomCrop(image_size, image_size),
        ]
    return [A.Resize(image_size, image_size)]


def get_train_transforms(image_size=256, crop=False):
    """Transforms for training set with augmentation."""
    return A.Compose([
        *_spatial_transform(image_size, crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.ColorJitter(p=1),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256, crop=False):
    """Transforms for validation set (no augmentation)."""
    return A.Compose([
        *_spatial_transform(image_size, crop),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size=256):
    """Transforms for inference."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def create_dataloaders(data_dir, batch_size=8, image_size=256, num_workers=4,
                       crop=False, balanced_crop=False, rare_classes=(1, 2),
                       balanced_min_fraction=0.05, balanced_attempts=10):
    """
    Creates DataLoaders for training and validation.

    Args:
        data_dir: data directory (train/, val/)
        batch_size: batch size
        image_size: image size
        num_workers: number of workers
        crop: if True, use native-resolution crops instead of resizing
              the whole image (better for small classes such as buildings).
        balanced_crop: if True (and crop), oversample training crops that contain
              rare classes so the model actually sees urban/water often enough.
        rare_classes: class ids treated as rare for balanced sampling.
        balanced_min_fraction: minimum fraction of rare-class pixels a crop must
              contain to be accepted immediately.
        balanced_attempts: how many random windows to try before falling back to
              the best one found.

    Validation always uses a deterministic center crop (when crop=True) so the
    metrics are stable across epochs instead of jumping with random windows.
    """
    train_dataset = SatelliteDataset(
        images_dir=os.path.join(data_dir, 'train', 'images'),
        masks_dir=os.path.join(data_dir, 'train', 'masks'),
        image_size=image_size,
        crop=crop,
        augment=True,
        balanced_crop=balanced_crop,
        rare_classes=rare_classes,
        balanced_min_fraction=balanced_min_fraction,
        balanced_attempts=balanced_attempts,
    )
    
    val_dataset = SatelliteDataset(
        images_dir=os.path.join(data_dir, 'val', 'images'),
        masks_dir=os.path.join(data_dir, 'val', 'masks'),
        image_size=image_size,
        crop=crop,
        augment=False,
        center_crop=crop,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def log_class_distribution(loader, num_classes=4, max_batches=20, class_names=None):
    """Print the per-class pixel share over the first `max_batches` batches.

    This shows what the model actually sees after cropping/sampling. With plain
    random crops urban/water are tiny; with balanced cropping their share should
    rise noticeably. Useful sanity check before committing to a long run.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    batches = 0
    for _, masks in loader:
        flat = masks.numpy().reshape(-1)
        counts += np.bincount(flat, minlength=num_classes)[:num_classes]
        batches += 1
        if batches >= max_batches:
            break

    total = int(counts.sum())
    print(f"Class pixel distribution over {batches} sampled batch(es):")
    for c in range(num_classes):
        name = class_names[c] if class_names else str(c)
        pct = 100.0 * counts[c] / total if total else 0.0
        print(f"    {name}: {pct:5.2f}%  ({int(counts[c])} px)")
    return counts


def create_demo_data(output_dir, num_samples=50, image_size=256):
    """
    Creates synthetic demo data for testing the pipeline.

    Generates simple images with random shapes:
    - Rectangles = urban areas (1)
    - Circles = water (2)
    - Irregular polygons = vegetation (3)
    """
    import random
    from PIL import ImageDraw

    for split in ['train', 'val']:
        images_dir = os.path.join(output_dir, split, 'images')
        masks_dir = os.path.join(output_dir, split, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        n = num_samples if split == 'train' else num_samples // 5

        for i in range(n):
            bg_color = (
                random.randint(100, 150),
                random.randint(100, 130),
                random.randint(80, 110)
            )
            image = Image.new('RGB', (image_size, image_size), bg_color)
            mask = Image.new('L', (image_size, image_size), 0)
            
            draw_img = ImageDraw.Draw(image)
            draw_mask = ImageDraw.Draw(mask)

            for _ in range(random.randint(3, 8)):
                shape_type = random.choice(['urban', 'water', 'vegetation'])
                x = random.randint(0, image_size - 50)
                y = random.randint(0, image_size - 50)
                w = random.randint(20, 80)
                h = random.randint(20, 80)
                
                if shape_type == 'urban':
                    color = (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))
                    draw_img.rectangle([x, y, x+w, y+h], fill=color)
                    draw_mask.rectangle([x, y, x+w, y+h], fill=1)
                    
                elif shape_type == 'water':
                    color = (random.randint(30, 80), random.randint(80, 150), random.randint(150, 220))
                    draw_img.ellipse([x, y, x+w, y+h], fill=color)
                    draw_mask.ellipse([x, y, x+w, y+h], fill=2)
                    
                else:  # vegetation
                    color = (random.randint(20, 80), random.randint(120, 200), random.randint(20, 80))
                    points = []
                    for _ in range(random.randint(5, 8)):
                        px = x + random.randint(0, w)
                        py = y + random.randint(0, h)
                        points.append((px, py))
                    if len(points) >= 3:
                        draw_img.polygon(points, fill=color)
                        draw_mask.polygon(points, fill=3)
            
            image.save(os.path.join(images_dir, f'sample_{i:04d}.png'))
            mask.save(os.path.join(masks_dir, f'sample_{i:04d}.png'))

        print(f"Created {n} samples in {split}/")


if __name__ == "__main__":
    print("Creating demo data...")
    create_demo_data("data", num_samples=50, image_size=256)
    print("Done!")

