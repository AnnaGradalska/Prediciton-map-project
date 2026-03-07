# -*- coding: utf-8 -*-
"""
Download and process DeepGlobe Land Cover Classification dataset.
https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

DeepGlobe has 7 classes (RGB colors in masks):
    - Urban land:      (0, 255, 255)   - cyan
    - Agriculture:     (255, 255, 0)   - yellow
    - Rangeland:       (255, 0, 255)   - magenta
    - Forest land:     (0, 255, 0)     - green
    - Water:           (0, 0, 255)     - blue
    - Barren land:     (255, 255, 255) - white
    - Unknown:         (0, 0, 0)       - black

Mapping to our 4 classes:
    0 - Other:         Agriculture, Rangeland, Barren, Unknown
    1 - Urban:         Urban land
    2 - Water:         Water
    3 - Vegetation:    Forest land
"""

import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


DEEPGLOBE_COLORS = {
    'urban':       (0, 255, 255),
    'agriculture': (255, 255, 0),
    'rangeland':   (255, 0, 255),
    'forest':      (0, 255, 0),
    'water':       (0, 0, 255),
    'barren':      (255, 255, 255),
    'unknown':     (0, 0, 0),
}

CLASS_MAPPING = {
    'urban':       1,  # Urban
    'agriculture': 3,  # Vegetation (crops)
    'rangeland':   3,  # Vegetation (rangeland)
    'forest':      3,  # Vegetation (forest)
    'water':       2,  # Water
    'barren':      0,  # Other (barren)
    'unknown':     0,  # Other
}


def download_dataset():
    """Download dataset from Kaggle."""
    try:
        import kagglehub
        print("Downloading DeepGlobe dataset from Kaggle...")
        path = kagglehub.dataset_download("balraj98/deepglobe-land-cover-classification-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        print("kagglehub not installed.")
        print("  Install: pip install kagglehub")
        print("  Then log in at kaggle.com -> Account -> Create New Token")
        return None
    except Exception as e:
        print(f"Download error: {e}")
        return None


def convert_mask(mask_rgb):
    """
    Convert DeepGlobe RGB mask to mask with values 0-3.

    Args:
        mask_rgb: numpy array (H, W, 3) RGB mask

    Returns:
        mask: numpy array (H, W) with values 0-3
    """
    h, w = mask_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for class_name, color in DEEPGLOBE_COLORS.items():
        matches = np.all(mask_rgb == color, axis=2)
        mask[matches] = CLASS_MAPPING[class_name]

    return mask


def process_dataset(source_path, output_path, train_ratio=0.8, max_samples=None):
    """
    Process DeepGlobe dataset to our format.

    Args:
        source_path: path to downloaded dataset
        output_path: output path (e.g. 'data')
        train_ratio: training data ratio (0.8 = 80% train, 20% val)
        max_samples: max number of samples (None = all)
    """
    print(f"\nProcessing dataset from: {source_path}")

    train_dir = None
    for root, dirs, files in os.walk(source_path):
        if 'train' in dirs:
            train_dir = os.path.join(root, 'train')
            break
        if any(f.endswith('_sat.jpg') for f in files):
            train_dir = root
            break

    if train_dir is None:
        print("Training data not found in dataset!")
        print(f"  Check structure: {source_path}")
        return False

    print(f"Found data in: {train_dir}")

    all_files = os.listdir(train_dir)
    sat_files = sorted([f for f in all_files if f.endswith('_sat.jpg')])

    print(f"Found {len(sat_files)} satellite images")

    if max_samples:
        sat_files = sat_files[:max_samples]
        print(f"  (limited to {max_samples} samples)")

    n_train = int(len(sat_files) * train_ratio)
    train_files = sat_files[:n_train]
    val_files = sat_files[n_train:]

    print(f"  Train: {len(train_files)}, Val: {len(val_files)}")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'masks'), exist_ok=True)

    for split, files in [('train', train_files), ('val', val_files)]:
        print(f"\nProcessing {split}...")

        for sat_file in tqdm(files, desc=split):
            base_name = sat_file.replace('_sat.jpg', '')
            sat_path = os.path.join(train_dir, sat_file)
            mask_path = os.path.join(train_dir, base_name + '_mask.png')

            if not os.path.exists(mask_path):
                print(f"  Missing mask for: {sat_file}")
                continue

            try:
                image = Image.open(sat_path).convert('RGB')
                mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
                mask = convert_mask(mask_rgb)
                output_name = base_name + '.png'
                image.save(os.path.join(output_path, split, 'images', output_name))
                Image.fromarray(mask).save(os.path.join(output_path, split, 'masks', output_name))
            except Exception as e:
                print(f"  Error processing {sat_file}: {e}")

    print(f"\nData saved to: {output_path}")
    print(f"   Train: {len(train_files)} images")
    print(f"   Val:   {len(val_files)} images")

    return True


def show_class_distribution(data_path):
    """Show class distribution in the data."""
    print("\nClass distribution:")

    class_names = {0: "Other", 1: "Urban", 2: "Water", 3: "Vegetation"}
    total_pixels = {0: 0, 1: 0, 2: 0, 3: 0}

    masks_dir = os.path.join(data_path, 'train', 'masks')
    if not os.path.exists(masks_dir):
        print("   No data to analyze")
        return

    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')][:100]

    for mask_file in tqdm(mask_files, desc="Analysis"):
        mask = np.array(Image.open(os.path.join(masks_dir, mask_file)))
        for c in range(4):
            total_pixels[c] += np.sum(mask == c)

    total = sum(total_pixels.values())
    print("\n   Class            | Percent")
    print("   -----------------|--------")
    for c, name in class_names.items():
        pct = (total_pixels[c] / total) * 100 if total > 0 else 0
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"   {name:15} | {pct:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(description='Download and process DeepGlobe dataset')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to already downloaded dataset (skip download)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training data ratio (default 0.8)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max number of samples (default: all)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download (use --source)')

    args = parser.parse_args()

    print("=" * 60)
    print("DeepGlobe Land Cover Dataset Downloader")
    print("=" * 60)

    if args.source:
        source_path = args.source
    elif args.skip_download:
        print("Provide --source when using --skip-download")
        return
    else:
        source_path = download_dataset()
        if source_path is None:
            return

    success = process_dataset(
        source_path,
        args.output,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples
    )

    if success:
        show_class_distribution(args.output)
        print("\n" + "=" * 60)
        print("Done. You can now train the model:")
        print(f"   python training/train.py --data-dir {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()

