"""
FIXED: Training Module with Proper Checkpoint Handling AND CORRECT VALIDATION
- All checkpoint paths are now consistent and properly handled
- Validation now uses memory bank retrieval (same as test) for accurate metrics
"""
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np


class Trainer:
    """Enhanced trainer with fixed checkpoint management and proper validation"""

    def __init__(self, model, criterion, config, device, save_dir: Path):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.scaler = GradScaler('cuda' if device == 'cuda' else 'cpu')

        # Training history
        self.history = {
            'train_loss': [],
            'val_top1': [],
            'val_top5': [],
            'learning_rate': []
        }

        self.best_val_top1 = 0.0
        self.best_epoch = 0
        self.current_stage = None

    def train_stage(
        self,
        train_loader,
        val_loader,
        class_list: List[str],
        memory_bank,  # ADDED: memory bank for validation
        stage_name: str,
        num_epochs: int = 100
    ) -> Dict:
        """
        Train for a specific stage

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_list: List of class names
            memory_bank: Memory bank for validation (will be updated each epoch)
            stage_name: Name of the training stage
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {stage_name.upper()}")
        print(f"Classes: {len(class_list)}, Epochs: {num_epochs}")
        print(f"{'='*80}")

        self.current_stage = stage_name
        self.best_val_top1 = 0.0
        self.best_epoch = 0

        # Reset history
        self.history = {
            'train_loss': [],
            'val_top1': [],
            'val_top5': [],
            'learning_rate': []
        }

        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.config.training.warmup_epochs:
                return (epoch + 1) / self.config.training.warmup_epochs
            else:
                progress = (epoch - self.config.training.warmup_epochs) / \
                          (num_epochs - self.config.training.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)

            # Train
            train_loss = self._train_epoch(train_loader, optimizer)

            # Update memory bank with current model embeddings
            # This ensures validation uses up-to-date prototypes
            self._update_memory_bank_for_validation(train_loader, class_list, memory_bank)

            # Validate using memory bank (same as test!)
            val_metrics = self._validate_epoch(val_loader, class_list, memory_bank)

            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_top1'].append(val_metrics['top1'])
            self.history['val_top5'].append(val_metrics['top5'])
            self.history['learning_rate'].append(current_lr)

            # Print metrics
            print(f"Loss: {train_loss:.4f} | Val Top-1: {val_metrics['top1']:.4f} | "
                  f"Val Top-5: {val_metrics['top5']:.4f} | LR: {current_lr:.6f}")

            # Save best model
            if val_metrics['top1'] > self.best_val_top1:
                improvement = val_metrics['top1'] - self.best_val_top1
                self.best_val_top1 = val_metrics['top1']
                self.best_epoch = epoch + 1
                self._save_checkpoint(stage_name, epoch + 1)
                print(f"✓ New best! Val Top-1: {self.best_val_top1:.4f} (+{improvement:.4f})")

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best Val Top-1: {self.best_val_top1:.4f} (Epoch {self.best_epoch})")
        print(f"{'='*80}")

        # Save training curves
        self._plot_training_curves(stage_name)
        self._save_history(stage_name)

        return self.history

    def _update_memory_bank_for_validation(self, train_loader, class_list, memory_bank):
        """
        Update memory bank with current epoch's embeddings
        This is crucial for validation to reflect actual test performance
        """
        self.model.eval()

        # Clear existing embeddings for these classes
        # (We want fresh embeddings from current model state)
        for class_name in class_list:
            if class_name in memory_bank.class_embeddings:
                memory_bank.class_embeddings[class_name] = []

        # Collect embeddings per class
        class_embeddings_dict = {cls: [] for cls in class_list}
        idx_to_class = {i: cls for i, cls in enumerate(class_list)}

        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                embeddings = self.model(images)

                # Group by class
                for emb, label in zip(embeddings, labels):
                    class_name = idx_to_class[label.item()]
                    class_embeddings_dict[class_name].append(emb)

        # Add to memory bank
        for class_name, embeddings_list in class_embeddings_dict.items():
            if len(embeddings_list) > 0:
                embeddings_tensor = torch.stack(embeddings_list)
                memory_bank.add_class(class_name, embeddings_tensor)

    def _train_epoch(self, train_loader, optimizer) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_norm
            )

            self.scaler.step(optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        return epoch_loss / num_batches

    @torch.no_grad()
    def _validate_epoch(self, val_loader, class_list, memory_bank) -> Dict:
        """
        FIXED: Validate using memory bank retrieval (SAME AS TEST!)
        This ensures validation metrics accurately reflect test performance
        """
        self.model.eval()

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        idx_to_class = {i: cls for i, cls in enumerate(class_list)}

        for images, labels in val_loader:
            images = images.to(self.device)
            embeddings = self.model(images)

            for emb, label in zip(embeddings, labels):
                # Use memory bank retrieval (SAME AS TEST!)
                result = memory_bank.retrieve(
                    emb,
                    use_knn=self.config.inference.use_knn,
                    k=self.config.inference.knn_k,
                    use_adaptive=self.config.inference.use_adaptive_threshold,
                    use_per_class_threshold=self.config.inference.use_per_class_threshold,
                    global_threshold=self.config.inference.rejection_threshold,
                    std_multiplier=self.config.inference.adaptive_threshold_std_mult
                )

                pred_class = result['class_name']
                true_class = idx_to_class[label.item()]

                # Top-1 accuracy
                if pred_class == true_class:
                    correct_top1 += 1

                # Top-5 from similarities
                sims = result.get('all_similarities', {})
                if sims:
                    top5 = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:5]
                    if true_class in top5:
                        correct_top5 += 1
                else:
                    # Fallback if no similarities
                    if pred_class == true_class:
                        correct_top5 += 1

                total += 1

        return {
            'top1': correct_top1 / total if total > 0 else 0.0,
            'top5': correct_top5 / total if total > 0 else 0.0
        }

    def _save_checkpoint(self, stage_name: str, epoch: int):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.config.model.embedding_dim,
            'best_val_top1': self.best_val_top1,
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'history': self.history
        }

        # Save to checkpoint directory
        checkpoint_path = self.checkpoint_dir / f'{stage_name}_best.pth'
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, stage_name: str) -> bool:
        """Load checkpoint - returns True if successful"""
        checkpoint_path = self.checkpoint_dir / f'{stage_name}_best.pth'

        if not checkpoint_path.exists():
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # For PyTorch 2.6+, use weights_only=False for now (safest for our checkpoints)
            # Our checkpoints contain numpy arrays and other complex types
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_top1 = checkpoint.get('best_val_top1', 0.0)
            print(f"✓ Loaded checkpoint from {checkpoint_path}")
            print(f"  Epoch: {checkpoint['epoch']}, Val Top-1: {self.best_val_top1:.4f}")
            return True
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            return False

    def _plot_training_curves(self, stage_name: str):
        """Plot and save training curves"""
        if len(self.history['train_loss']) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Val Top-1
        axes[0, 1].plot(epochs, self.history['val_top1'], color='green', linewidth=2)
        axes[0, 1].axhline(y=self.best_val_top1, color='r', linestyle='--',
                          label=f'Best: {self.best_val_top1:.4f}')
        axes[0, 1].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Top-1 Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Val Top-5
        axes[1, 0].plot(epochs, self.history['val_top5'], color='orange', linewidth=2)
        axes[1, 0].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Top-5 Accuracy')
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rate'], color='purple', linewidth=2)
        axes[1, 1].axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Training Curves - {stage_name.upper()}', fontsize=14)
        plt.tight_layout()

        save_path = self.save_dir / f'training_curves_{stage_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Training curves saved: {save_path}")

    def _save_history(self, stage_name: str):
        """Save training history"""
        history_with_meta = {
            'stage': stage_name,
            'best_epoch': self.best_epoch,
            'best_val_top1': self.best_val_top1,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        save_path = self.save_dir / f'training_history_{stage_name}.json'
        with open(save_path, 'w') as f:
            json.dump(history_with_meta, f, indent=2, default=str)

        print(f"✓ Training history saved: {save_path}")