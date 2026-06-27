# -*- coding: utf-8 -*-
"""
Visual sanity check for the dataset.

Dumps a grid of random image / mask / overlay triples into a single PNG so you
can eyeball whether the labels are correct BEFORE committing to a long training
run. This catches the most expensive hidden problems:

- masks shifted/misaligned vs the image,
- urban mislabeled as other (or vice versa),
- whole classes missing or merged,
- nonsense mask edges.

It also prints the per-image class pixel share so you can spot images that are
pure vegetation (which dominate DeepGlobe).

Usage:
    python training/inspect_data.py --data-dir data_deepglobe --split train --num 20
    python training/inspect_data.py --data-dir data_deepglobe --split val --num 12 --output val_check.png
"""

import os
import sys
import argparse
import random

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import CLASS_NAMES, CLASS_COLORS

VALID_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def colorize_mask(mask):
    """Map a (H, W) class-id mask to an (H, W, 3) RGB image using CLASS_COLORS."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    return rgb


def find_mask_path(masks_dir, stem):
    for ext in VALID_EXT:
        path = os.path.join(masks_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def load_mask(path, shape):
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def class_fractions(mask, num_classes=4):
    total = mask.size
    counts = np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]
    return counts / total if total else counts


def main():
    parser = argparse.ArgumentParser(description="Dump random image/mask pairs for visual QA.")
    parser.add_argument('--data-dir', type=str, default='data_deepglobe',
                        help='Dataset root containing <split>/images and <split>/masks.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Which split to inspect.')
    parser.add_argument('--num', type=int, default=20, help='Number of pairs to show.')
    parser.add_argument('--thumb', type=int, default=256, help='Thumbnail size (px) per cell.')
    parser.add_argument('--alpha', type=float, default=0.45, help='Overlay opacity for the mask.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG (default: inspect_<split>.png in cwd).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible sampling.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    images_dir = os.path.join(args.data_dir, args.split, 'images')
    masks_dir = os.path.join(args.data_dir, args.split, 'masks')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images dir at '{images_dir}'. Check --data-dir / --split.")

    images = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(VALID_EXT))
    if not images:
        raise FileNotFoundError(f"No images found in '{images_dir}'.")

    n = min(args.num, len(images))
    sample = random.sample(images, n)

    thumb = args.thumb
    cols = 3  # image | mask | overlay
    header = 18
    grid = Image.new('RGB', (cols * thumb, n * (thumb + header)), (20, 20, 28))

    print(f"Inspecting {n} random pairs from {args.split} ({len(images)} available)")
    print(f"Legend: {', '.join(f'{i}={name}' for i, name in CLASS_NAMES.items())}")

    for row, img_name in enumerate(sample):
        stem = os.path.splitext(img_name)[0]
        image = Image.open(os.path.join(images_dir, img_name)).convert('RGB')
        mask = load_mask(find_mask_path(masks_dir, stem), (image.height, image.width))

        fr = class_fractions(mask)
        frac_str = ' '.join(f'{CLASS_NAMES[i]}={fr[i] * 100:4.1f}%' for i in range(len(fr)))
        print(f"  {img_name}: {frac_str}")

        img_t = image.resize((thumb, thumb))
        mask_rgb = Image.fromarray(colorize_mask(mask)).resize((thumb, thumb), Image.NEAREST)
        overlay = Image.blend(img_t, mask_rgb, args.alpha)

        y = row * (thumb + header) + header
        grid.paste(img_t, (0, y))
        grid.paste(mask_rgb, (thumb, y))
        grid.paste(overlay, (2 * thumb, y))

    output = args.output or f'inspect_{args.split}.png'
    grid.save(output)
    print(f"\nSaved grid to: {os.path.abspath(output)}")
    print("Columns: [original | mask | overlay]")


if __name__ == "__main__":
    main()
