# -*- coding: utf-8 -*-
"""
Training script for satellite image segmentation model.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet, UNetResNet, CLASS_NAMES
from models.deeplabv3 import get_deeplabv3_resnet101
from training.dataset import create_dataloaders, create_demo_data, log_class_distribution

# DeepGlobe class frequencies (Other 11%, Urban 7.9%, Water 3%, Vegetation 78.1%)
DEEPGLOBE_CLASS_FREQS = [0.11, 0.079, 0.03, 0.781]


def get_class_weights(freqs, num_classes=4):
    """Weights inverse to frequency: weight_c = 1 / (freq_c * num_classes), then normalized."""
    weights = [1.0 / (f * num_classes) if f > 0 else 1.0 for f in freqs]
    s = sum(weights)
    return [w / s * num_classes for w in weights]


def make_colab_downloader():
    """Return a callback that downloads a file to the user's computer from Colab.

    Uses google.colab.files.download, which streams the file to the browser's
    download folder. Returns None if not running inside a Colab kernel (e.g. local
    runs or `!python` subprocess), in which case saving stays local-only.
    """
    try:
        from google.colab import files  # type: ignore
    except Exception:
        print("  [download] google.colab not available - model stays on the local "
              "filesystem only (run training in a Colab notebook cell to enable "
              "auto-download).")
        return None

    def _download(path):
        try:
            print(f"  [download] Sending {os.path.basename(path)} to your computer...")
            files.download(path)
        except Exception as e:  # pragma: no cover - depends on Colab frontend
            print(f"  [download] Could not download {path}: {e}")

    return _download


def dice_coefficient(pred, target, num_classes=4, smooth=1e-6):
    """Compute Dice coefficient per class."""
    dice_scores = []
    
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    for c in range(num_classes):
        pred_c = pred_one_hot[:, c]
        target_c = target_one_hot[:, c]
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    return dice_scores


def iou_score(pred, target, num_classes=4, smooth=1e-6):
    """Compute IoU (Intersection over Union) per class."""
    iou_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    return iou_scores


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation. Optional class weights (E2)."""

    def __init__(self, num_classes=4, smooth=1e-6, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_per_class = dice_per_class.mean(dim=0)

        if self.class_weights is not None:
            w = self.class_weights.to(pred.device)
            dice_weighted = (w * dice_per_class).sum() / w.sum()
        else:
            dice_weighted = dice_per_class.mean()

        return 1 - dice_weighted


class CombinedLoss(nn.Module):
    """Combination of Cross Entropy and Dice loss. Optional class weights (E2)."""

    def __init__(self, num_classes=4, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.ce = nn.CrossEntropyLoss(weight=weight_tensor)
            self.dice = DiceLoss(num_classes, class_weights=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()
            self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_logits(outputs):
    """Extract logit tensor from model output.
    U-Net/UNetResNet return a tensor directly.
    DeepLabV3+ (torchvision) returns a dict {'out': tensor, 'aux': tensor}.
    """
    if isinstance(outputs, dict):
        return outputs["out"]
    return outputs


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    all_dice = [0, 0, 0, 0]

    pbar = tqdm(train_loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = get_logits(model(images))
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        dice = dice_coefficient(preds, masks)
        for i in range(4):
            all_dice[i] += dice[i]
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n = len(train_loader)
    avg_loss = total_loss / n
    avg_dice = [d / n for d in all_dice]
    
    return avg_loss, avg_dice


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_dice = [0, 0, 0, 0]
    all_iou = [0, 0, 0, 0]
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = get_logits(model(images))
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            dice = dice_coefficient(preds, masks)
            iou = iou_score(preds, masks)
            
            for i in range(4):
                all_dice[i] += dice[i]
                all_iou[i] += iou[i]
    
    n = len(val_loader)
    avg_loss = total_loss / n
    avg_dice = [d / n for d in all_dice]
    avg_iou = [i / n for i in all_iou]
    
    return avg_loss, avg_dice, avg_iou


def train(args, on_best_save=None):
    """Main training function.

    on_best_save: optional callback(path) invoked every time a new best model is
    saved. Used to auto-download the model to the user's computer on Colab.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if on_best_save is None and getattr(args, 'download', False):
        on_best_save = make_colab_downloader()

    if not os.path.exists(os.path.join(args.data_dir, 'train', 'images')):
        if not args.demo:
            raise FileNotFoundError(
                f"No training data at '{os.path.join(args.data_dir, 'train', 'images')}'.\n"
                "Check --data-dir (e.g. point it to the real DeepGlobe folder).\n"
                "To intentionally train on synthetic demo data, pass --demo."
            )
        print("No training data. Creating demo data (--demo)...")
        create_demo_data(args.data_dir, num_samples=args.demo_samples, image_size=args.image_size)

    rare_classes = tuple(int(c) for c in args.rare_classes.split(',')) if args.rare_classes else (1, 2)
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        crop=args.crop,
        balanced_crop=args.balanced_crop,
        rare_classes=rare_classes,
        balanced_min_fraction=args.balanced_min_fraction,
        balanced_attempts=args.balanced_attempts,
    )

    # Sanity check: how much of each class the model actually sees after
    # cropping/sampling. Compare with/without --balanced-crop.
    if args.crop:
        print(f"Balanced crop: {args.balanced_crop} "
              f"(rare={rare_classes}, min_fraction={args.balanced_min_fraction}, "
              f"attempts={args.balanced_attempts})")
    log_class_distribution(train_loader, num_classes=4, max_batches=20,
                           class_names=CLASS_NAMES)

    if args.model == "unet_resnet":
        model = UNetResNet(n_channels=3, n_classes=4, pretrained=True)
        if args.freeze_encoder > 0:
            model.freeze_encoder(True)
            print(f"Encoder frozen for first {args.freeze_encoder} epochs (E3).")
    elif args.model == "deeplabv3":
        # DeepLabV3+ with ResNet-101 backbone (torchvision), 4 output classes
        model = get_deeplabv3_resnet101(num_classes=4, pretrained=True)
    else:
        model = UNet(n_channels=3, n_classes=4, bilinear=True)
    model = model.to(device)

    print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.class_weights:
        class_weights = get_class_weights(DEEPGLOBE_CLASS_FREQS)
        print(f"Class weights (E2): {[f'{w:.3f}' for w in class_weights]}")
        criterion = CombinedLoss(num_classes=4, ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)
    else:
        criterion = CombinedLoss(num_classes=4, ce_weight=0.5, dice_weight=0.5)
    criterion = criterion.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_dice = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        if args.model == "unet_resnet" and args.freeze_encoder > 0 and epoch == args.freeze_encoder:
            model.freeze_encoder(False)
            print("Encoder unfrozen; fine-tuning full model.")

        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        mean_train_dice = np.mean(train_dice)
        mean_val_dice = np.mean(val_dice)
        mean_val_iou = np.mean(val_iou)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {mean_train_dice:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {mean_val_dice:.4f} | Val IoU: {mean_val_iou:.4f}")
        
        print(f"  Dice per class:")
        for i, name in CLASS_NAMES.items():
            print(f"    {name}: {val_dice[i]:.4f} (IoU: {val_iou[i]:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dice = mean_val_dice
            # Save best model into <output_root>/<model_name>/<model_name>.pth.
            # Point --output-root at a Drive folder to keep the model safe even if
            # the Colab session is interrupted (best model is written every time it
            # improves, not only at the end).
            best_model_dir = os.path.join(args.output_root, args.model_name)
            os.makedirs(best_model_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_dir, f"{args.model_name}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f}) to {best_model_path}")
            if on_best_save is not None:
                on_best_save(best_model_path)

        if (epoch + 1) % args.save_every == 0:
            checkpoint_dir = os.path.join(args.output_root, args.model_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print("Training finished.")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Best model stored in: {os.path.join(args.output_root, args.model_name, args.model_name + '.pth')}")


def main():
    parser = argparse.ArgumentParser(description='Train satellite segmentation model')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Training data directory')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model')
    parser.add_argument('--output-root', type=str, default='trained_models',
                        help='Root folder where the model is saved. Point this at a '
                             'Google Drive folder on Colab (e.g. '
                             '/content/drive/MyDrive/trained_models) so the best model '
                             'is written straight to Drive and survives a session reset.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for AdamW')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--demo-samples', type=int, default=100,
                        help='Number of demo samples (if no data)')
    parser.add_argument('--class-weights', action='store_true',
                        help='Use class weights (E2): inverse to DeepGlobe frequency')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unet_resnet', 'deeplabv3'],
                        help='Model: unet (baseline), unet_resnet (E3, ResNet-50 encoder) or deeplabv3 (E5, DeepLabV3+ ResNet-101)')
    parser.add_argument('--freeze-encoder', type=int, default=0,
                        help='Freeze encoder for first N epochs (E3). 0 = no freeze.')
    parser.add_argument('--crop', action='store_true',
                        help='Train on native-resolution random crops instead of '
                             'resizing whole images (better for small classes like '
                             'buildings). Use with tiled inference in the app. '
                             'Validation uses a deterministic center crop.')
    parser.add_argument('--balanced-crop', action='store_true',
                        help='With --crop, oversample training crops that contain '
                             'rare classes (urban/water) so the model stops '
                             'defaulting to vegetation. Strongly recommended.')
    parser.add_argument('--rare-classes', type=str, default='1,2',
                        help='Comma-separated class ids treated as rare for '
                             'balanced cropping (default 1,2 = urban,water).')
    parser.add_argument('--balanced-min-fraction', type=float, default=0.05,
                        help='Min fraction of rare-class pixels a crop must have '
                             'to be accepted by balanced sampling (default 0.05).')
    parser.add_argument('--balanced-attempts', type=int, default=10,
                        help='How many random windows to try before falling back '
                             'to the best one (default 10).')
    parser.add_argument('--demo', action='store_true',
                        help='Allow generating synthetic demo data when --data-dir '
                             'has no real data. Without this flag, missing data is an error.')
    parser.add_argument('--download', action='store_true',
                        help='On Colab, download the best model to your computer every '
                             'time it improves (no Drive needed). Requires running '
                             'training inside a notebook cell, not via "!python".')

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

