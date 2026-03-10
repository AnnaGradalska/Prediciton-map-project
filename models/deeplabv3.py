import torch.nn as nn

from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)


def get_deeplabv3_resnet101(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Create DeepLabV3+ style model with ResNet-101 backbone for semantic segmentation.

    Uses torchvision implementation and replaces the classifier head to predict `num_classes`.
    """
    weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet101(weights=weights)

    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

    return model

