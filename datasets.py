"""
Dataset Module with Enhanced Transforms Integration
"""
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from PIL import Image
from typing import Optional, List
from augmentations import get_train_transforms, get_val_transforms


class TrafficSignDataset(Dataset):
    """Traffic sign dataset loader"""

    def __init__(
        self,
        data_root: Path,
        split: str = 'train',
        transform=None,
        class_list: Optional[List[str]] = None
    ):
        """
        Args:
            data_root: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            transform: Transforms to apply (if None, uses default)
            class_list: List of class names to include (None = all classes)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # Get split directory
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Get all class directories
        all_classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

        # Filter classes if class_list provided
        if class_list is not None:
            self.classes = [c for c in class_list if c in all_classes]
        else:
            self.classes = all_classes

        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = split_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            print(f"WARNING: No samples found for {split} split with {len(self.classes)} classes")

        print(f"{split.capitalize()} dataset: {len(self.samples)} samples, "
              f"{len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self):
        """Get number of samples per class"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


def get_dataloader(
    dataset: TrafficSignDataset,
    config,
    is_training: bool = True,
    use_balanced_sampler: bool = False
):
    """
    Create dataloader with optional balanced sampling

    Args:
        dataset: TrafficSignDataset
        config: SystemConfig object
        is_training: Whether this is training data
        use_balanced_sampler: Use weighted sampler for class balance

    Returns:
        DataLoader
    """
    batch_size = config.training.batch_size
    num_workers = config.data.num_workers

    if is_training and use_balanced_sampler:
        # Calculate weights for balanced sampling
        class_counts = dataset.get_class_counts()

        # Weight = 1 / count for each sample
        weights = []
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            weight = 1.0 / class_counts[class_name]
            weights.append(weight)

        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_training
        )

    return loader