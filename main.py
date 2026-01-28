"""
Complete System Orchestrator - FIXED AND ENHANCED
Zero errors, production-ready implementation

Usage:
    python main.py --run-all          # Full pipeline
    python main.py --stage 1          # Individual stages
"""
import torch
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Import modules
from config import SystemConfig
from backbone import EmbeddingModel
from losses import SupervisedContrastiveLoss
from memory_bank import EnhancedMemoryBank
from datasets import TrafficSignDataset, get_dataloader
from augmentations import get_train_transforms, get_val_transforms
from training import Trainer
from incremental import IncrementalLearner


class TrafficSignSystem:
    """Complete orchestrator for the improved pipeline"""

    def __init__(self, data_root: Path = Path("prepared_dataset/ready_training")):
        # Initialize configuration
        self.config = SystemConfig()
        self.config.data.data_root = data_root

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print("\nInitializing model...")
        self.model = EmbeddingModel(
            backbone_name=self.config.model.backbone,
            pretrained=self.config.model.pretrained,
            hidden_dim=self.config.model.hidden_dim,
            embedding_dim=self.config.model.embedding_dim,
            projection_layers=self.config.model.projection_layers,
            dropout=self.config.model.dropout,
            use_batch_norm=self.config.model.use_batch_norm
        ).to(self.device)

        # Print model info
        param_info = self.model.get_num_parameters()
        print(f"Model parameters: {param_info['total']:,} total, "
              f"{param_info['trainable']:,} trainable")

        # Initialize components
        self.criterion = SupervisedContrastiveLoss(
            temperature=self.config.training.temperature
        )

        self.memory_bank = EnhancedMemoryBank(
            self.config.model.embedding_dim,
            self.device
        )

        # Initialize pipelines
        self.trainer = Trainer(
            self.model, self.criterion, self.config,
            self.device, self.results_dir
        )

        self.incremental_learner = IncrementalLearner(
            self.model, self.memory_bank, self.config, self.device
        )

        # Load class splits
        self.class_splits = self._get_class_splits()

        print(f"\nSystem initialized")
        print(f"Results directory: {self.results_dir}")

    def _get_class_splits(self):
        """Get class splits for each stage"""
        train_dir = self.config.data.data_root / 'train'

        if not train_dir.exists():
            print(f"ERROR: Training directory not found: {train_dir}")
            return None

        all_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"\nFound {len(all_classes)} classes in training set")

        # Stage 1: First 100 classes
        base_classes = all_classes[:100]

        # Stage 2: Next 50 classes
        incremental_classes = all_classes[100:150] if len(all_classes) >= 150 else []

        splits = {
            'stage_1': base_classes,
            'stage_2_new': incremental_classes,
            'stage_2_all': base_classes + incremental_classes
        }

        print(f"Stage 1 (base): {len(base_classes)} classes")
        print(f"Stage 2 (new): {len(incremental_classes)} classes")
        print(f"Stage 2 (total): {len(splits['stage_2_all'])} classes")

        return splits

    def run_stage_1(self, num_epochs: int = 100):
        """Stage 1: Train on 100 base classes"""
        print("\n" + "=" * 80)
        print("STAGE 1: TRAINING ON 100 BASE CLASSES")
        print("=" * 80)

        class_list = self.class_splits['stage_1']

        # Create datasets
        train_transform = get_train_transforms(self.config)
        val_transform = get_val_transforms(self.config)

        train_dataset = TrafficSignDataset(
            self.config.data.data_root, 'train',
            transform=train_transform, class_list=class_list
        )
        val_dataset = TrafficSignDataset(
            self.config.data.data_root, 'val',
            transform=val_transform, class_list=class_list
        )

        # Create loaders
        train_loader = get_dataloader(
            train_dataset, self.config,
            is_training=True, use_balanced_sampler=True
        )
        val_loader = get_dataloader(
            val_dataset, self.config,
            is_training=False, use_balanced_sampler=False
        )

        # Train
        print(f"\nTraining for {num_epochs} epochs...")
        history = self.trainer.train_stage(
            train_loader, val_loader, class_list,
            self.memory_bank,  # ← ADD THIS LINE
            'stage_1', num_epochs
        )

        # Extract embeddings and build memory bank
        print("\nBuilding memory bank...")
        self.incremental_learner.extract_and_store_embeddings(
            class_list, 'stage_1'
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_transform = get_val_transforms(self.config)
        test_dataset = TrafficSignDataset(
            self.config.data.data_root, 'test',
            transform=test_transform, class_list=class_list
        )

        # Check if all classes are present
        if len(test_dataset.classes) < len(class_list):
            missing_classes = set(class_list) - set(test_dataset.classes)
            print(f"⚠ WARNING: {len(missing_classes)} classes missing from test set:")
            print(f"  Missing: {list(missing_classes)[:5]}...")
            print(f"  This is OK - evaluation will use {len(test_dataset.classes)} classes")

        test_loader = get_dataloader(
            test_dataset, self.config,
            is_training=False, use_balanced_sampler=False
        )

        results = self._evaluate_with_memory_bank(
            test_loader, class_list, 'stage_1'
        )

        # Freeze backbone for incremental learning
        if self.config.model.freeze_backbone_after_base:
            self.model.freeze_backbone()
            print("\n✓ Backbone frozen for incremental learning")

        return results

    def run_stage_2(self):
        """Stage 2: Incremental learning (+50 classes)"""
        print("\n" + "=" * 80)
        print("STAGE 2: INCREMENTAL LEARNING (+50 CLASSES)")
        print("=" * 80)

        # Try to load stage 1 checkpoint from multiple locations
        checkpoint_loaded = False
        checkpoint_locations = [
            # Current results directory
            self.trainer.checkpoint_dir / 'stage_1_best.pth',
            # Global checkpoints directory
            Path('checkpoints/stage_1_best.pth'),
            # Most recent results directory
            None  # Will search
        ]

        # Search in all results directories for most recent stage 1
        if not checkpoint_locations[0].exists() and not checkpoint_locations[1].exists():
            results_dirs = sorted(Path('results').glob('*/checkpoints/stage_1_best.pth'))
            if results_dirs:
                checkpoint_locations.append(results_dirs[-1])  # Most recent

        for checkpoint_path in checkpoint_locations:
            if checkpoint_path and checkpoint_path.exists():
                print(f"Found Stage 1 checkpoint: {checkpoint_path}")
                try:
                    # Use weights_only=False for compatibility with all checkpoint types
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ Loaded Stage 1 checkpoint")
                    print(f"  Epoch: {checkpoint['epoch']}, Val Top-1: {checkpoint.get('best_val_top1', 0):.4f}")
                    checkpoint_loaded = True
                    break
                except Exception as e:
                    print(f"⚠ Failed to load {checkpoint_path}: {e}")
                    continue

        if not checkpoint_loaded:
            print("\n❌ ERROR: Cannot load Stage 1 checkpoint!")
            print("\nStage 1 must be completed before Stage 2.")
            print("Please run: python main.py --stage 1 --epochs 100")
            return None

        # Load stage 1 memory bank
        memory_bank_paths = [
            Path(self.config.embedding_dir) / 'memory_bank_stage_1.pkl',
            Path('embeddings/memory_bank_stage_1.pkl')
        ]

        memory_bank_loaded = False
        for memory_bank_path in memory_bank_paths:
            if memory_bank_path.exists():
                try:
                    self.memory_bank.load(memory_bank_path)
                    print(f"✓ Loaded stage 1 memory bank: {len(self.memory_bank)} classes")
                    memory_bank_loaded = True
                    break
                except Exception as e:
                    print(f"⚠ Failed to load {memory_bank_path}: {e}")
                    continue

        if not memory_bank_loaded:
            print(f"⚠ WARNING: Memory bank not found")
            print("Rebuilding from checkpoint...")
            self.incremental_learner.extract_and_store_embeddings(
                self.class_splits['stage_1'], 'stage_1'
            )
            # Try loading again
            memory_bank_path = Path(self.config.embedding_dir) / 'memory_bank_stage_1.pkl'
            if memory_bank_path.exists():
                self.memory_bank.load(memory_bank_path)

        old_classes = self.class_splits['stage_1']
        new_classes = self.class_splits['stage_2_new']
        all_classes = self.class_splits['stage_2_all']

        if not new_classes:
            print("No new classes available for Stage 2")
            return None

        # Add new classes (no retraining)
        print(f"\nAdding {len(new_classes)} new classes...")
        self.incremental_learner.add_new_classes(new_classes, 'stage_2')

        # Evaluate
        print("\nEvaluating on test set...")
        test_transform = get_val_transforms(self.config)
        test_dataset = TrafficSignDataset(
            self.config.data.data_root, 'test',
            transform=test_transform, class_list=all_classes
        )
        test_loader = get_dataloader(
            test_dataset, self.config,
            is_training=False, use_balanced_sampler=False
        )

        results = self._evaluate_with_memory_bank(
            test_loader, all_classes, 'stage_2',
            old_classes=old_classes, new_classes=new_classes
        )

        return results

    @torch.no_grad()
    def _evaluate_with_memory_bank(
        self,
        test_loader,
        class_list,
        stage_name,
        old_classes=None,
        new_classes=None
    ):
        """Evaluate using memory bank with enhanced retrieval"""
        print(f"\nEvaluating {stage_name.upper()}...")
        print(f"Memory bank has {len(self.memory_bank)} classes")
        print(f"Test set has {len(class_list)} classes")

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_top5 = []

        # Debug: track unknown predictions
        unknown_count = 0
        correct_count = 0
        total_count = 0

        idx_to_class = {i: cls for i, cls in enumerate(class_list)}

        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(self.device)
            embeddings = self.model(images)

            for emb, label in zip(embeddings, labels):
                total_count += 1

                # Get prediction from memory bank
                result = self.memory_bank.retrieve(
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

                # Debug tracking
                if pred_class == 'unknown':
                    unknown_count += 1
                if pred_class == true_class:
                    correct_count += 1

                all_predictions.append(pred_class)
                all_labels.append(true_class)

                # Get top-5 from similarities
                sims = result.get('all_similarities', {})
                if sims:
                    top5 = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:5]
                else:
                    top5 = [pred_class]
                all_top5.append(top5)

        # Print debug info
        print(f"\nDebug Info:")
        print(f"  Total predictions: {total_count}")
        print(f"  Correct predictions: {correct_count} ({100*correct_count/total_count:.1f}%)")
        print(f"  Unknown predictions: {unknown_count} ({100*unknown_count/total_count:.1f}%)")
        print(f"  Using {'k-NN' if self.config.inference.use_knn else 'prototype'} retrieval")

        # Compute metrics
        results = self._compute_metrics(
            all_labels, all_predictions, all_top5,
            class_list, old_classes, new_classes, stage_name
        )

        # Save results
        results_path = self.results_dir / f'results_{stage_name}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Results saved: {results_path}")

        return results

    def _compute_metrics(
        self,
        true_labels,
        predictions,
        top5_preds,
        class_list,
        old_classes,
        new_classes,
        stage_name
    ):
        """Compute comprehensive metrics"""
        results = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'num_classes': len(class_list),
            'num_samples': len(true_labels)
        }

        if len(true_labels) == 0:
            print("⚠ WARNING: No samples to evaluate!")
            results.update({
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'mean_per_class_accuracy': 0.0
            })
            return results

        # Top-1 accuracy
        correct_top1 = sum([p == l for p, l in zip(predictions, true_labels)])
        results['top1_accuracy'] = correct_top1 / len(true_labels)

        # Top-5 accuracy
        correct_top5 = sum([l in preds for l, preds in zip(true_labels, top5_preds)])
        results['top5_accuracy'] = correct_top5 / len(true_labels)

        # Per-class accuracy
        per_class_acc = {}
        for cls in class_list:
            cls_mask = [l == cls for l in true_labels]
            if sum(cls_mask) > 0:
                cls_correct = sum([p == l for p, l, m in
                                   zip(predictions, true_labels, cls_mask) if m])
                per_class_acc[cls] = cls_correct / sum(cls_mask)

        results['per_class_accuracy'] = per_class_acc
        results['mean_per_class_accuracy'] = (
            float(np.mean(list(per_class_acc.values()))) if per_class_acc else 0.0
        )

        # Old/New class metrics for stage 2
        if old_classes:
            old_mask = [l in old_classes for l in true_labels]
            if sum(old_mask) > 0:
                old_correct = sum([p == l for p, l, m in
                                   zip(predictions, true_labels, old_mask) if m])
                results['old_classes_top1'] = old_correct / sum(old_mask)

        if new_classes:
            new_mask = [l in new_classes for l in true_labels]
            if sum(new_mask) > 0:
                new_correct = sum([p == l for p, l, m in
                                   zip(predictions, true_labels, new_mask) if m])
                results['new_classes_top1'] = new_correct / sum(new_mask)

        # Print summary
        print(f"\n{stage_name.upper()} Results:")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"  Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.4f}")

        if 'old_classes_top1' in results:
            print(f"  Old Classes Top-1: {results['old_classes_top1']:.4f}")
        if 'new_classes_top1' in results:
            print(f"  New Classes Top-1: {results['new_classes_top1']:.4f}")

        return results

    def run_complete_pipeline(self, num_epochs_stage1: int = 100):
        """Run complete pipeline: Stage 1 -> Stage 2"""
        print("\n" + "=" * 80)
        print("COMPLETE PIPELINE EXECUTION")
        print("=" * 80)
        print(f"Results will be saved to: {self.results_dir}")

        all_results = {}

        # Stage 1
        print("\n[1/2] Running Stage 1...")
        all_results['stage_1'] = self.run_stage_1(num_epochs_stage1)

        # Stage 2
        print("\n[2/2] Running Stage 2...")
        all_results['stage_2'] = self.run_stage_2()

        # Save complete results
        summary_path = self.results_dir / 'complete_results_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"\nResults Summary:")
        print(f"  Stage 1 Top-1: {all_results['stage_1']['top1_accuracy']:.4f}")
        if all_results['stage_2']:
            print(f"  Stage 2 Top-1: {all_results['stage_2']['top1_accuracy']:.4f}")
            if 'old_classes_top1' in all_results['stage_2']:
                print(f"    Old Classes: {all_results['stage_2']['old_classes_top1']:.4f}")
            if 'new_classes_top1' in all_results['stage_2']:
                print(f"    New Classes: {all_results['stage_2']['new_classes_top1']:.4f}")

        print(f"\nAll results saved to: {self.results_dir}")

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Few-Shot Incremental Traffic Sign Recognition System"
    )

    parser.add_argument('--run-all', action='store_true',
                        help='Run complete pipeline (stages 1, 2)')
    parser.add_argument('--stage', type=int, choices=[1, 2],
                        help='Run specific stage only')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for stage 1 training')
    parser.add_argument('--data-root', type=str,
                        default='prepared_dataset/ready_training',
                        help='Path to prepared dataset')

    args = parser.parse_args()

    # Initialize system
    system = TrafficSignSystem(Path(args.data_root))

    # Run pipeline
    if args.run_all:
        system.run_complete_pipeline(args.epochs)

    elif args.stage == 1:
        system.run_stage_1(args.epochs)

    elif args.stage == 2:
        system.run_stage_2()

    else:
        print("Please specify --run-all or --stage N")
        print("Use --help for more information")


if __name__ == "__main__":
    main()