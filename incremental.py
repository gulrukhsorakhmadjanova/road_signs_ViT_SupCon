"""
Enhanced Incremental Learning Module
Works with the enhanced memory bank for better few-shot performance
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List
from datasets import TrafficSignDataset, get_dataloader
from augmentations import get_val_transforms


class IncrementalLearner:
    """Handles incremental learning for new classes"""

    def __init__(self, model, memory_bank, config, device):
        self.model = model
        self.memory_bank = memory_bank
        self.config = config
        self.device = device

    def extract_and_store_embeddings(
        self,
        class_list: List[str],
        stage_name: str
    ):
        """
        Extract ALL embeddings from training data and store in memory bank
        (Not just prototypes - stores all embeddings for k-NN)
        """
        print(f"\nExtracting embeddings for {len(class_list)} classes...")

        # Create dataset with NO augmentation
        transform = get_val_transforms(self.config)
        train_dataset = TrafficSignDataset(
            self.config.data.data_root,
            split='train',
            transform=transform,
            class_list=class_list
        )

        if len(train_dataset) == 0:
            print(f"⚠ WARNING: No samples found in training set!")
            return

        print(f"Train dataset: {len(train_dataset)} samples, "
              f"{len(train_dataset.classes)} classes")

        # Create dataloader
        train_loader = get_dataloader(
            train_dataset,
            self.config,
            is_training=False,
            use_balanced_sampler=False
        )

        # Extract embeddings
        self.model.eval()
        class_embeddings_dict = {}

        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting embeddings"):
                images = images.to(self.device)
                embeddings = self.model(images)

                # Group by class
                for emb, label in zip(embeddings, labels):
                    class_name = train_dataset.classes[label.item()]

                    if class_name not in class_embeddings_dict:
                        class_embeddings_dict[class_name] = []

                    class_embeddings_dict[class_name].append(emb.cpu())

        # Add to memory bank
        print(f"\nAdding embeddings to memory bank...")
        for class_name in class_list:
            if class_name in class_embeddings_dict:
                embeddings = torch.stack(class_embeddings_dict[class_name])
                self.memory_bank.add_class(class_name, embeddings)
            else:
                print(f"⚠ WARNING: No samples found for class '{class_name}'")

        # Save memory bank
        embeddings_dir = Path(self.config.embedding_dir)
        embeddings_dir.mkdir(exist_ok=True, parents=True)

        memory_bank_path = embeddings_dir / f'memory_bank_{stage_name}.pkl'
        self.memory_bank.save(memory_bank_path)

        print(f"✓ Memory bank saved: {len(self.memory_bank)} classes -> {memory_bank_path}")

    def add_new_classes(self, new_classes: List[str], stage_name: str):
        """
        Add new classes to the system (incremental learning)
        Extracts and stores ALL embeddings for k-NN
        """
        print(f"\nAdding {len(new_classes)} new classes incrementally...")

        # Extract embeddings for new classes
        transform = get_val_transforms(self.config)
        train_dataset = TrafficSignDataset(
            self.config.data.data_root,
            split='train',
            transform=transform,
            class_list=new_classes
        )

        if len(train_dataset) == 0:
            print(f"⚠ WARNING: No samples found for new classes!")
            return

        print(f"Train dataset: {len(train_dataset)} samples, "
              f"{len(train_dataset.classes)} classes")

        train_loader = get_dataloader(
            train_dataset,
            self.config,
            is_training=False,
            use_balanced_sampler=False
        )

        # Extract embeddings
        self.model.eval()
        class_embeddings_dict = {}

        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting new class embeddings"):
                images = images.to(self.device)
                embeddings = self.model(images)

                for emb, label in zip(embeddings, labels):
                    class_name = train_dataset.classes[label.item()]

                    if class_name not in class_embeddings_dict:
                        class_embeddings_dict[class_name] = []

                    class_embeddings_dict[class_name].append(emb.cpu())

        # Add to memory bank
        print(f"\nAdding new classes to memory bank...")
        added_count = 0
        for class_name in new_classes:
            if class_name in class_embeddings_dict:
                embeddings = torch.stack(class_embeddings_dict[class_name])
                self.memory_bank.add_class(class_name, embeddings)
                added_count += 1
            else:
                print(f"⚠ WARNING: No samples found for class '{class_name}'")

        print(f"✓ Added {added_count}/{len(new_classes)} new classes")
        print(f"Memory bank now contains {len(self.memory_bank)} classes")

        # Save updated memory bank
        embeddings_dir = Path(self.config.embedding_dir)
        memory_bank_path = embeddings_dir / f'memory_bank_{stage_name}.pkl'
        self.memory_bank.save(memory_bank_path)