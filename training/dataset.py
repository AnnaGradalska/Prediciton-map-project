# -*- coding: utf-8 -*-
"""
Dataset for training satellite image segmentation model.
"""

import os
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

    def __init__(self, images_dir, masks_dir, transform=None, image_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size

        self.images = sorted([f for f in os.listdir(images_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        print(f"Found {len(self.images)} images in {images_dir}")
    
    def __len__(self):
        return len(self.images)
    
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

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


def get_train_transforms(image_size=256):
    """Transforms for training set with augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
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


def get_val_transforms(image_size=256):
    """Transforms for validation set (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
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


def create_dataloaders(data_dir, batch_size=8, image_size=256, num_workers=4):
    """
    Creates DataLoaders for training and validation.

    Args:
        data_dir: data directory (train/, val/)
        batch_size: batch size
        image_size: image size
        num_workers: number of workers
    """
    train_dataset = SatelliteDataset(
        images_dir=os.path.join(data_dir, 'train', 'images'),
        masks_dir=os.path.join(data_dir, 'train', 'masks'),
        transform=get_train_transforms(image_size),
        image_size=image_size
    )
    
    val_dataset = SatelliteDataset(
        images_dir=os.path.join(data_dir, 'val', 'images'),
        masks_dir=os.path.join(data_dir, 'val', 'masks'),
        transform=get_val_transforms(image_size),
        image_size=image_size
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

