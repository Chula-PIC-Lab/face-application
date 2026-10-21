"""
Models Package for Facial Emotion Recognition

This package contains:
- EfficientFace: Lightweight CNN combining ShuffleNetV2 backbone, Local Feature Extractor,
  and Attention Modulator.
- Modulator: Channel and Spatial attention mechanism.
- ResNet: Standard deep residual networks used as baseline and teacher network for distillation.
"""

from models.EfficientFace import EfficientFace, efficient_face
from models.modulator import Modulator, Channel, Spatial
from models.resnet import resnet18, resnet34, resnet50

__all__ = [
    "EfficientFace",
    "efficient_face",
    "Modulator",
    "Channel",
    "Spatial",
    "resnet18",
    "resnet34",
    "resnet50",
]
