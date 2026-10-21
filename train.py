"""
train.py - Facial Emotion Recognition Training, Testing, Inference & Fine-Tuning with PyTorch Lightning

This module provides a complete PyTorch Lightning pipeline for training, testing,
evaluating, and fine-tuning EfficientFace (and ResNet baselines) on facial expression datasets.

Features:
1. PyTorch Lightning Architecture (LightningModule & LightningDataModule).
2. Supports training from scratch, resuming, and fine-tuning.
3. Fine-tuning on custom student-captured images (from webcam capture sessions in app.py).
4. Universal Checkpoint Loading: Supports BOTH plain PyTorch (.pth.tar, .pth) and Lightning (.ckpt).
5. Checkpoint Converter: Convert existing non-lightning (.pth.tar) checkpoints to native Lightning (.ckpt).
6. Testing & evaluation with accuracy metrics.
7. Single-image and batch inference.
8. Export to ONNX format for deployment back into app.py.

Contribution Areas for Students:
Look for blocks marked with:
    ##################
    # TODO: ...
    ##################
"""

import argparse
import glob
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# Compatible with both `pytorch_lightning` and `lightning`
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
except ImportError:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required to run train.py. "
            "Please install it using: pip install lightning (or pip install pytorch-lightning)"
        )

from models.EfficientFace import efficient_face
from models.resnet import resnet18, resnet50

# Standard 7 facial emotion classes
EMOTIONS = [
    "0_Neutral",
    "1_Happiness",
    "2_Sadness",
    "3_Surprise",
    "4_Fear",
    "5_Disgust",
    "6_Anger",
]

# Aliases to parse student-captured filenames robustly
EMOTION_NAME_TO_INDEX = {
    "neutral": 0,
    "happiness": 1,
    "hapiness": 1,
    "happy": 1,
    "sadness": 2,
    "sad": 2,
    "surprise": 3,
    "surprised": 3,
    "fear": 4,
    "fearful": 4,
    "disgust": 5,
    "disgusted": 5,
    "anger": 6,
    "angry": 6,
}

# Image normalization constants (AffectNet / RAF-DB / EfficientFace)
MEAN = [0.57535914, 0.44928582, 0.40079932]
STD = [0.20735591, 0.18981615, 0.18132027]


# -----------------------------------------------------------------------------
# 1. Custom Dataset for Student Captured Images
# -----------------------------------------------------------------------------
class CapturedFaceDataset(Dataset):
    """
    Dataset to load images captured by students from the webcam (e.g. in app.py).
    
    Supports two common directory arrangements:
    1. Flat directory with filenames formatted as:
       - `capture_<timestamp>_<class_index>_<emotion_name>.jpg`
         (e.g., `capture_1792406467769_0_Neutral.jpg`, `capture_1792406512183_1_Hapiness.jpg`)
       - `<class_index>_*.jpg`
       - `<emotion_name>_*.jpg`
    2. Subfolder directory:
       - `captured_images/0_Neutral/xxx.jpg`, `captured_images/1_Happiness/xxx.jpg`, etc.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        if not os.path.exists(root_dir):
            print(f"[Warning] Captured images directory not found: '{root_dir}'")
            return

        self._scan_images()
        self._print_class_summary()

    def _parse_label_from_filename(self, filename: str) -> Optional[int]:
        stem = os.path.splitext(os.path.basename(filename))[0]

        # Pattern: capture_<timestamp>_<index>_<emotion>
        # e.g., capture_1792406467769_0_Neutral
        parts = stem.split("_")
        for part in parts:
            if part.isdigit() and 0 <= int(part) < len(EMOTIONS):
                return int(part)
            lower_part = part.lower()
            if lower_part in EMOTION_NAME_TO_INDEX:
                return EMOTION_NAME_TO_INDEX[lower_part]

        return None

    def _scan_images(self):
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

        # 1. Check for subfolder structure (ImageFolder style)
        subdirs = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith(".")
        ]

        if subdirs:
            for subdir in subdirs:
                label = None
                # Check if subdir name contains class index or emotion
                for i, emo in enumerate(EMOTIONS):
                    if emo.lower() in subdir.lower() or f"{i}_" in subdir:
                        label = i
                        break
                if label is None:
                    lower_dir = subdir.lower()
                    for emo_name, idx in EMOTION_NAME_TO_INDEX.items():
                        if emo_name in lower_dir:
                            label = idx
                            break

                if label is not None:
                    folder_path = os.path.join(self.root_dir, subdir)
                    for f in os.listdir(folder_path):
                        if f.lower().endswith(valid_extensions):
                            self.samples.append((os.path.join(folder_path, f), label))

        # 2. Check for flat files in root_dir
        for f in os.listdir(self.root_dir):
            if f.lower().endswith(valid_extensions) and not f.startswith("."):
                full_path = os.path.join(self.root_dir, f)
                label = self._parse_label_from_filename(f)
                if label is not None:
                    self.samples.append((full_path, label))

    def _print_class_summary(self):
        counts = {i: 0 for i in range(len(EMOTIONS))}
        for _, lbl in self.samples:
            counts[lbl] += 1
        print("\n" + "=" * 50)
        print(f"CapturedFaceDataset loaded from: {self.root_dir}")
        print(f"Total samples found: {len(self.samples)}")
        for i, emo in enumerate(EMOTIONS):
            print(f"  [{i}] {emo}: {counts[i]} images")
        print("=" * 50 + "\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# -----------------------------------------------------------------------------
# 2. DataModule
# -----------------------------------------------------------------------------
class EmotionDataModule(pl.LightningDataModule):
    """
    DataModule handling data loading, augmentations, and train/val/test splits.
    
    Supports:
    - Benchmark datasets (FER2013 with train/ and test/ folders).
    - Custom captured student images for fine-tuning.
    - Blended datasets (combining standard dataset with student-captured images).
    """

    def __init__(
        self,
        data_dir: Optional[str] = "dataset/FER2013",
        custom_data_dir: Optional[str] = "captured_images",
        use_custom_only: bool = False,
        batch_size: int = 32,
        num_workers: int = 2,
        image_size: int = 224,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.custom_data_dir = custom_data_dir
        self.use_custom_only = use_custom_only
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        ##################
        # TODO: Student Contribution - Data Augmentations
        # Experiment with stronger data augmentations to improve generalization on webcam images:
        # - transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        # - transforms.RandomRotation(degrees=15)
        # - transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        # - transforms.RandomErasing(p=0.2)
        ##################
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop((self.image_size, self.image_size), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def setup(self, stage: Optional[str] = None):
        if self.use_custom_only:
            # Fine-tune only on student-captured images
            if not self.custom_data_dir or not os.path.isdir(self.custom_data_dir):
                raise ValueError(f"Custom data directory '{self.custom_data_dir}' does not exist.")

            full_dataset = CapturedFaceDataset(self.custom_data_dir, transform=self.train_transforms)
            if len(full_dataset) == 0:
                raise ValueError(
                    f"No valid emotion images found in '{self.custom_data_dir}'. "
                    "Please capture images using app.py or place them in captured_images/"
                )

            val_size = max(1, int(len(full_dataset) * self.val_split))
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            self.test_dataset = self.val_dataset

        else:
            # Benchmark dataset (FER2013)
            train_path = os.path.join(self.data_dir, "train") if self.data_dir else None
            test_path = os.path.join(self.data_dir, "test") if self.data_dir else None

            if train_path and os.path.isdir(train_path):
                base_train = datasets.ImageFolder(train_path, transform=self.train_transforms)

                # Optionally blend student captured images into the training set
                if self.custom_data_dir and os.path.isdir(self.custom_data_dir):
                    custom_dataset = CapturedFaceDataset(self.custom_data_dir, transform=self.train_transforms)
                    if len(custom_dataset) > 0:
                        print(f"Blending {len(custom_dataset)} student-captured images into training set!")
                        self.train_dataset = torch.utils.data.ConcatDataset([base_train, custom_dataset])
                    else:
                        self.train_dataset = base_train
                else:
                    self.train_dataset = base_train

            if test_path and os.path.isdir(test_path):
                self.val_dataset = datasets.ImageFolder(test_path, transform=self.val_transforms)
                self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Did you call setup()?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


# -----------------------------------------------------------------------------
# 3. PyTorch Lightning Module with Universal Checkpoint Loading
# -----------------------------------------------------------------------------
class EmotionClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for Facial Emotion Classification.
    
    Supports:
    - EfficientFace backbone (default) and ResNet baselines.
    - Pretrained checkpoint loading (from BOTH PyTorch .pth.tar and Lightning .ckpt).
    - Backbone layer freezing for fine-tuning on custom images.
    - Training, validation, testing, and inference step methods.
    - ONNX model export.
    """

    def __init__(
        self,
        model_name: str = "efficient_face",
        num_classes: int = 7,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        pretrained_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone

        # Build model architecture
        self.model = self._build_model(model_name, num_classes)

        # Load weights if checkpoint provided
        if pretrained_checkpoint and os.path.isfile(pretrained_checkpoint):
            self._load_pretrained_weights(pretrained_checkpoint)

        # Handle backbone freezing for fine-tuning
        if freeze_backbone:
            self._freeze_backbone_parameters()

        ##################
        # TODO: Student Contribution - Loss Functions
        # Experiment with advanced loss functions to handle emotion class imbalance:
        # 1. Class-weighted CrossEntropyLoss:
        #    weights = torch.tensor([...]).to(self.device)
        #    self.criterion = nn.CrossEntropyLoss(weight=weights)
        # 2. Focal Loss: focuses training on hard negative examples.
        # 3. Label Smoothing CrossEntropy: reduces overconfident predictions.
        ##################
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self, model_name: str, num_classes: int) -> nn.Module:
        if model_name == "efficient_face":
            return efficient_face(num_classes=num_classes)
        elif model_name == "resnet18":
            return resnet18(num_classes=num_classes)
        elif model_name == "resnet50":
            return resnet50(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose 'efficient_face', 'resnet18', or 'resnet50'.")

    def _load_pretrained_weights(self, checkpoint_path: str):
        print(f"=> Loading weights from: '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Clean prefixes (DataParallel 'module.' or Lightning 'model.')
        cleaned = {}
        for k, v in state_dict.items():
            k_clean = k
            while k_clean.startswith("module.") or k_clean.startswith("model."):
                if k_clean.startswith("module."):
                    k_clean = k_clean[7:]
                elif k_clean.startswith("model."):
                    k_clean = k_clean[6:]
            cleaned[k_clean] = v

        # Filter out mismatching head weights if classes differ
        model_dict = self.model.state_dict()
        filtered = {
            k: v for k, v in cleaned.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        self.model.load_state_dict(filtered, strict=False)
        print(f"=> Successfully loaded {len(filtered)} / {len(model_dict)} layers from checkpoint.")

    def _freeze_backbone_parameters(self):
        """Freezes all layers except the final classification head (fc)."""
        print("=> Freezing backbone layers. Only classifier head will be trained.")
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == targets).float().mean() * 100.0
        return loss, acc

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        loss, acc = self._compute_metrics(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        loss, acc = self._compute_metrics(logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        loss, acc = self._compute_metrics(logits, targets)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(images)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return {"probs": probs, "preds": preds}

    def configure_optimizers(self):
        ##################
        # TODO: Student Contribution - Optimizer & Schedulers
        # Experiment with:
        # 1. Differential Learning Rates: lower LR for backbone, higher LR for classification head.
        #    params = [
        #        {"params": [p for n, p in self.model.named_parameters() if "fc" not in n and p.requires_grad], "lr": self.lr * 0.1},
        #        {"params": self.model.fc.parameters(), "lr": self.lr},
        #    ]
        # 2. Schedulers: CosineAnnealingWarmRestarts, OneCycleLR, or ReduceLROnPlateau.
        ##################
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(10, self.trainer.max_epochs or 30), eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def export_to_onnx(self, output_path: str, input_size: int = 224):
        """Exports the trained model to ONNX format for deployment."""
        self.eval()
        dummy_input = torch.randn(1, 3, input_size, input_size, device=self.device)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"=> Successfully exported ONNX model to: '{output_path}'")

    @classmethod
    def load_model(
        cls,
        checkpoint_path: str,
        model_name: str = "efficient_face",
        num_classes: int = 7,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "EmotionClassifier":
        """
        Universal Checkpoint Loader:
        Seamlessly loads BOTH:
          1. PyTorch Lightning checkpoints (.ckpt)
          2. Standard PyTorch checkpoints (.pth.tar, .pth, .pt from main.py)
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: '{checkpoint_path}'")

        map_loc = device if device is not None else "cpu"
        is_lightning_ckpt = False

        try:
            raw = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(raw, dict) and ("pytorch-lightning_version" in raw or ("epoch" in raw and "global_step" in raw)):
                is_lightning_ckpt = True
        except Exception:
            pass

        if is_lightning_ckpt:
            try:
                model = cls.load_from_checkpoint(
                    checkpoint_path,
                    map_location=map_loc,
                    strict=False,
                    **kwargs,
                )
                print(f"=> Successfully loaded PyTorch Lightning checkpoint: '{checkpoint_path}'")
                if device is not None:
                    model.to(device)
                return model
            except Exception as e:
                print(f"[Notice] Lightning load_from_checkpoint fell back to weights loader: {e}")

        # Standard PyTorch Checkpoint (.pth.tar, .pth, .pt)
        print(f"=> Loading standard PyTorch checkpoint into LightningModule: '{checkpoint_path}'")
        model = cls(
            model_name=model_name,
            num_classes=num_classes,
            pretrained_checkpoint=checkpoint_path,
            **kwargs,
        )
        if device is not None:
            model.to(device)
        return model


# -----------------------------------------------------------------------------
# 4. Checkpoint Conversion Utility (.pth.tar -> .ckpt)
# -----------------------------------------------------------------------------
def convert_to_lightning_checkpoint(
    source_path: str,
    output_path: str,
    model_name: str = "efficient_face",
    num_classes: int = 7,
):
    """
    Converts a standard PyTorch checkpoint (.pth.tar / .pth) into a native
    PyTorch Lightning checkpoint (.ckpt) so students can use standard
    Lightning utilities.
    """
    print(f"\n=> Converting PyTorch checkpoint: '{source_path}' -> Lightning checkpoint: '{output_path}'")
    model = EmotionClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained_checkpoint=source_path,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(output_path)
    print(f"=> Conversion complete! Saved native Lightning checkpoint to: '{output_path}'\n")


# -----------------------------------------------------------------------------
# 5. Standalone Inference Helper
# -----------------------------------------------------------------------------
def infer_image(
    image_path: str,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Runs emotion inference on a single image file."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load model using Universal Checkpoint Loader
    classifier = EmotionClassifier.load_model(checkpoint_path, device=device)
    classifier.eval()

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_emotion = EMOTIONS[pred_idx]

    print("\n" + "=" * 45)
    print(f"Inference Result for: {image_path}")
    print(f"Dominant Emotion: {pred_emotion} ({probs[pred_idx] * 100:.2f}%)")
    print("-" * 45)
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo:<15}: {probs[i] * 100:6.2f}%")
    print("=" * 45 + "\n")

    return {emo: float(probs[i]) for i, emo in enumerate(EMOTIONS)}


# -----------------------------------------------------------------------------
# 6. CLI & Execution Entry Point
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning FER Training, Testing, Inference & Fine-Tuning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "fine_tune", "test", "infer", "export_onnx", "convert"],
        help="Execution mode: train, fine_tune (on captured images), test, infer, export_onnx, or convert (.pth.tar -> .ckpt)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficient_face",
        choices=["efficient_face", "resnet18", "resnet50"],
        help="Backbone model architecture",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/FER2013",
        help="Path to dataset directory containing train/ and test/ folders",
    )
    parser.add_argument(
        "--custom_data_dir",
        type=str,
        default="./captured_images",
        help="Path to captured images for fine-tuning",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoint/[09-30]-[05-21]-model_best.pth.tar",
        help="Path to initial weights or model checkpoint (.pth.tar or .ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoint",
        help="Directory to save output checkpoints",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default="./checkpoint/model_best_lightning.ckpt",
        help="Output file path when running '--mode convert' to save .ckpt",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze backbone and only train the classifier head (recommended for fine-tuning)",
    )
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Image file path for single image inference (--mode infer)",
    )
    parser.add_argument(
        "--onnx_output",
        type=str,
        default="./models/efficient_face_student.onnx",
        help="Path to save exported ONNX model (--mode export_onnx)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device configuration
    accelerator = "gpu" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    devices = 1

    # -------------------------------------------------------------
    # Mode: Convert Checkpoint (.pth.tar -> .ckpt)
    # -------------------------------------------------------------
    if args.mode == "convert":
        convert_to_lightning_checkpoint(
            source_path=args.checkpoint_path,
            output_path=args.output_ckpt,
            model_name=args.model_name,
            num_classes=len(EMOTIONS),
        )
        return

    # -------------------------------------------------------------
    # Mode: Single Image Inference
    # -------------------------------------------------------------
    if args.mode == "infer":
        if not args.image_path:
            sample_candidates = glob.glob(os.path.join(args.custom_data_dir, "*.jpg"))
            if sample_candidates:
                args.image_path = sample_candidates[0]
                print(f"No --image_path specified. Using sample image: {args.image_path}")
            else:
                print("Error: Please provide --image_path for inference.")
                sys.exit(1)

        infer_image(args.image_path, args.checkpoint_path)
        return

    # -------------------------------------------------------------
    # Mode: Export to ONNX
    # -------------------------------------------------------------
    if args.mode == "export_onnx":
        print(f"=> Exporting checkpoint '{args.checkpoint_path}' to ONNX...")
        model = EmotionClassifier.load_model(
            checkpoint_path=args.checkpoint_path,
            model_name=args.model_name,
            num_classes=len(EMOTIONS),
        )
        model.export_to_onnx(args.onnx_output)
        return

    # -------------------------------------------------------------
    # Setup DataModule
    # -------------------------------------------------------------
    is_custom_only = (args.mode == "fine_tune")

    datamodule = EmotionDataModule(
        data_dir=args.data_dir,
        custom_data_dir=args.custom_data_dir,
        use_custom_only=is_custom_only,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # -------------------------------------------------------------
    # Mode: Test
    # -------------------------------------------------------------
    if args.mode == "test":
        print(f"=> Evaluating checkpoint: {args.checkpoint_path}")
        model = EmotionClassifier.load_model(
            checkpoint_path=args.checkpoint_path,
            model_name=args.model_name,
            num_classes=len(EMOTIONS),
        )

        trainer = pl.Trainer(accelerator=accelerator, devices=devices)
        datamodule.setup(stage="test")
        trainer.test(model, datamodule=datamodule)
        return

    # -------------------------------------------------------------
    # Mode: Train / Fine-Tune
    # -------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"Starting {args.mode.upper()} Pipeline")
    print(f"  Model:            {args.model_name}")
    print(f"  Mode:             {args.mode}")
    print(f"  Accelerator:      {accelerator}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Freeze Backbone:  {args.freeze_backbone or is_custom_only}")
    print("=" * 55 + "\n")

    # If fine-tuning, default freeze_backbone to True unless specified otherwise
    freeze_bb = args.freeze_backbone or is_custom_only

    model = EmotionClassifier(
        model_name=args.model_name,
        num_classes=len(EMOTIONS),
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone=freeze_bb,
        pretrained_checkpoint=args.checkpoint_path if os.path.isfile(args.checkpoint_path) else None,
    )

    # Checkpoint callback
    ckpt_name_prefix = "finetuned" if is_custom_only else "model"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{ckpt_name_prefix}_best_{{epoch:02d}}_{{val_acc:.2f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=10,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
    )

    # Run training
    trainer.fit(model, datamodule=datamodule)

    print(f"\n=> Training completed!")
    print(f"=> Best checkpoint saved at: {checkpoint_callback.best_model_path}")

    # Optional: Automatically test best model after training
    if datamodule.test_dataset is not None:
        print("\n=> Evaluating best model on test set...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
