"""
Enhanced Evaluation Pipeline with Stage-based Assessment
CLIP-COMPATIBLE VERSION: Uses CLIP normalization values
Supports Stage 0 (pretrained), Stage 1 (100 classes), Stage 2 (incremental)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class StageEvaluator:
    """Comprehensive evaluation for different training stages"""

    def __init__(self, model, memory_bank, config, device, results_dir: Path):
        self.model = model
        self.memory_bank = memory_bank
        self.config = config
        self.device = device
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Transform for evaluation - CLIP compatible
        self.eval_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])

    @torch.no_grad()
    def evaluate_stage_0_pretrained(self, train_dataset, test_dataset,
                                    class_list: List[str],
                                    stage_name: str = 'stage_0') -> Dict:
        """
        Stage 0: Evaluate pretrained backbone (no fine-tuning)
        Uses only the backbone features + simple classifier

        Args:
            train_dataset: Training dataset to build prototypes from
            test_dataset: Test dataset to evaluate on
            class_list: List of class names
            stage_name: Name of this stage
        """
        print(f"\n{'=' * 80}")
        print(f"STAGE 0 EVALUATION: Pretrained Backbone Only")
        print(f"{'=' * 80}")

        self.model.eval()

        # 1. Build prototypes from TRAINING set
        print("Building prototypes from training features...")
        train_prototypes = self._build_prototypes_from_data(
            train_dataset, class_list
        )

        print(f"Built {len(train_prototypes)} prototypes from training data")

        # 2. Extract features from TEST set
        print("Extracting test features...")
        all_embeddings = []
        all_labels = []
        all_images = []

        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        for images, labels in tqdm(test_loader, desc="Extracting test features"):
            images = images.to(self.device)

            # Get embeddings from model
            embeddings = self.model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = np.array(all_labels)

        print(f"Extracted features from {len(all_labels)} test samples")

        # 3. Compute predictions
        predictions, similarities, top5_preds = self._compute_predictions_topk(
            all_embeddings, train_prototypes, class_list
        )

        # Convert label indices to class names
        idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
        true_label_names = [idx_to_class[label] for label in all_labels]

        # 4. Compute metrics
        results = self._compute_metrics(
            np.array(true_label_names), predictions, top5_preds, class_list, stage_name
        )

        # 5. Visualizations
        self._save_confusion_matrix(
            true_label_names, predictions, class_list, stage_name
        )

        self._save_sample_predictions(
            all_images[:20], true_label_names[:20],
            top5_preds[:20], class_list, stage_name
        )

        # 6. Save results
        self._save_results_json(results, stage_name)

        return results

    @torch.no_grad()
    def evaluate_stage_with_memory_bank(self, test_loader, class_list: List[str],
                                        old_classes: Optional[List[str]] = None,
                                        new_classes: Optional[List[str]] = None,
                                        stage_name: str = 'stage_1') -> Dict:
        """
        Evaluate using memory bank (for trained stages)
        Computes metrics for all classes, old classes, and new classes separately
        """
        print(f"\n{'=' * 80}")
        print(f"{stage_name.upper()} EVALUATION: {len(class_list)} classes")
        if old_classes:
            print(f"Old classes: {len(old_classes)}, New classes: {len(new_classes)}")
        print(f"{'=' * 80}")

        self.model.eval()

        # Collect predictions
        all_embeddings = []
        all_labels = []
        all_predictions = []
        all_similarities = []
        all_images = []

        idx_to_class = {i: cls for i, cls in enumerate(class_list)}

        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(self.device)
            embeddings = self.model(images)

            all_embeddings.append(embeddings.cpu())

            for emb, label in zip(embeddings, labels):
                # Get prediction from memory bank
                result = self.memory_bank.retrieve(
                    emb,
                    use_adaptive=self.config.inference.use_adaptive_threshold,
                    global_threshold=self.config.inference.rejection_threshold,
                    std_multiplier=self.config.inference.adaptive_threshold_std_mult
                )

                all_predictions.append(result['class_name'])
                all_labels.append(idx_to_class[label.item()])
                all_similarities.append(result['all_similarities'])

            all_images.extend(images.cpu())

            if len(all_images) >= 100:
                break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Compute top-k predictions
        top5_preds = self._compute_topk_from_similarities(
            all_similarities, k=5
        )

        # Compute metrics
        results = self._compute_metrics_with_splits(
            all_labels, all_predictions, top5_preds,
            class_list, old_classes, new_classes, stage_name
        )

        # Visualizations
        self._save_confusion_matrix(
            all_labels, all_predictions, class_list, stage_name,
            max_classes=50  # Limit for readability
        )

        self._save_sample_predictions(
            all_images[:20], all_labels[:20],
            top5_preds[:20], class_list, stage_name
        )

        # Save results
        self._save_results_json(results, stage_name)

        return results

    def _build_prototypes_from_data(self, dataset, class_list: List[str]) -> Dict:
        """Build prototypes from dataset (for stage 0)"""
        class_embeddings = defaultdict(list)

        print(f"Building prototypes from {len(dataset)} samples...")

        # Use ALL training samples (not just 1000)
        # Or if too many, use stratified sampling
        max_samples_per_class = 100

        # Group samples by class
        class_samples = defaultdict(list)
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            class_name = class_list[label]
            class_samples[class_name].append(idx)

        # Sample from each class
        selected_indices = []
        for class_name, indices in class_samples.items():
            n_samples = min(max_samples_per_class, len(indices))
            sampled = np.random.choice(indices, n_samples, replace=False)
            selected_indices.extend(sampled)

        print(f"Using {len(selected_indices)} samples to build prototypes")

        for idx in tqdm(selected_indices, desc="Building prototypes"):
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model(img).cpu().numpy()[0]

            class_name = class_list[label]
            class_embeddings[class_name].append(emb)

        # Compute prototypes
        prototypes = {}
        for class_name, embeddings in class_embeddings.items():
            prototype = np.mean(embeddings, axis=0)
            prototype = prototype / np.linalg.norm(prototype)
            prototypes[class_name] = torch.tensor(prototype).to(self.device)

        print(f"Built prototypes for {len(prototypes)} classes")

        # Check coverage
        missing = set(class_list) - set(prototypes.keys())
        if missing:
            print(f"WARNING: No prototypes built for {len(missing)} classes: {list(missing)[:5]}")

        return prototypes

    def _compute_predictions_topk(self, embeddings: torch.Tensor,
                                  prototypes: Dict, class_list: List[str],
                                  k: int = 5) -> Tuple:
        """Compute predictions and top-k for given embeddings"""
        predictions = []
        similarities_list = []
        top5_preds = []

        for emb in embeddings:
            emb = emb.to(self.device)

            # Compute similarities
            sims = {}
            for class_name, proto in prototypes.items():
                sim = torch.dot(emb, proto).item()
                sims[class_name] = sim

            # Top-1
            best_class = max(sims, key=sims.get)
            predictions.append(best_class)
            similarities_list.append(sims)

            # Top-k
            sorted_classes = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)
            top5_preds.append(sorted_classes[:k])

        return predictions, similarities_list, top5_preds

    def _compute_topk_from_similarities(self, similarities_list: List[Dict],
                                        k: int = 5) -> List[List[str]]:
        """Extract top-k predictions from similarity dictionaries"""
        topk_preds = []

        for sims in similarities_list:
            sorted_classes = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)
            topk_preds.append(sorted_classes[:k])

        return topk_preds

    def _compute_metrics(self, true_labels: np.ndarray,
                         predictions: List[str],
                         top5_preds: List[List[str]],
                         class_list: List[str],
                         stage_name: str) -> Dict:
        """Compute comprehensive metrics"""
        results = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'num_classes': len(class_list),
            'num_samples': len(true_labels)
        }

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
        results['mean_per_class_accuracy'] = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

        print(f"\n{stage_name.upper()} Results:")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"  Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.4f}")

        return results

    def _compute_metrics_with_splits(self, true_labels: List[str],
                                     predictions: List[str],
                                     top5_preds: List[List[str]],
                                     class_list: List[str],
                                     old_classes: Optional[List[str]],
                                     new_classes: Optional[List[str]],
                                     stage_name: str) -> Dict:
        """Compute metrics with old/new class splits"""
        results = self._compute_metrics(
            np.array(true_labels), predictions, top5_preds, class_list, stage_name
        )

        # Old class metrics
        if old_classes:
            old_mask = [l in old_classes for l in true_labels]
            if sum(old_mask) > 0:
                old_correct_top1 = sum([p == l for p, l, m in
                                        zip(predictions, true_labels, old_mask) if m])
                results['old_classes_top1'] = old_correct_top1 / sum(old_mask)

                old_correct_top5 = sum([l in preds for l, preds, m in
                                        zip(true_labels, top5_preds, old_mask) if m])
                results['old_classes_top5'] = old_correct_top5 / sum(old_mask)
            else:
                results['old_classes_top1'] = 0.0
                results['old_classes_top5'] = 0.0

        # New class metrics
        if new_classes:
            new_mask = [l in new_classes for l in true_labels]
            if sum(new_mask) > 0:
                new_correct_top1 = sum([p == l for p, l, m in
                                        zip(predictions, true_labels, new_mask) if m])
                results['new_classes_top1'] = new_correct_top1 / sum(new_mask)

                new_correct_top5 = sum([l in preds for l, preds, m in
                                        zip(true_labels, top5_preds, new_mask) if m])
                results['new_classes_top5'] = new_correct_top5 / sum(new_mask)
            else:
                results['new_classes_top1'] = 0.0
                results['new_classes_top5'] = 0.0

        # Print split metrics if available
        if 'old_classes_top1' in results and 'new_classes_top1' in results:
            print(f"  Old Classes Top-1: {results['old_classes_top1']:.4f}")
            print(f"  New Classes Top-1: {results['new_classes_top1']:.4f}")
        elif 'old_classes_top1' in results:
            print(f"  Old Classes Top-1: {results['old_classes_top1']:.4f}")
        elif 'new_classes_top1' in results:
            print(f"  New Classes Top-1: {results['new_classes_top1']:.4f}")

        return results

    def _save_confusion_matrix(self, true_labels, predictions,
                               class_list: List[str], stage_name: str,
                               max_classes: int = 50):
        """Save confusion matrix visualization"""
        # Limit classes for readability
        if len(class_list) > max_classes:
            print(f"  Skipping confusion matrix ({len(class_list)} > {max_classes} classes)")
            return

        cm = confusion_matrix(true_labels, predictions, labels=class_list)

        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        fig, ax = plt.subplots(figsize=(max(12, len(class_list) // 3),
                                        max(10, len(class_list) // 3)))

        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=class_list, yticklabels=class_list,
                    cbar_kws={'label': 'Normalized Frequency'}, ax=ax)

        ax.set_title(f'Confusion Matrix - {stage_name.upper()}', fontsize=14, pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        save_path = self.results_dir / f'confusion_matrix_{stage_name}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  Confusion matrix saved: {save_path}")

    def _save_sample_predictions(self, images: List[torch.Tensor],
                                 true_labels: List[str],
                                 top5_preds: List[List[str]],
                                 class_list: List[str],
                                 stage_name: str,
                                 n_samples: int = 20):
        """Save sample predictions with top-5 probabilities"""
        n_samples = min(n_samples, len(images))

        # Create grid
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        # Denormalize images - USE CLIP NORMALIZATION
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)

        for idx in range(n_samples):
            ax = axes[idx]

            # Denormalize image
            img = images[idx] * std + mean
            img = torch.clamp(img, 0, 1)
            img_np = img.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.axis('off')

            # Add predictions
            true_label = true_labels[idx]
            top5 = top5_preds[idx]

            title = f"True: {true_label[:20]}\n"
            title += "Top-5:\n"
            for i, pred in enumerate(top5, 1):
                marker = "✓" if pred == true_label else "✗"
                title += f"{i}. {pred[:15]} {marker}\n"

            ax.set_title(title, fontsize=8, ha='left')

        # Hide remaining axes
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Sample Predictions - {stage_name.upper()}',
                     fontsize=14, y=0.995)
        plt.tight_layout()

        save_path = self.results_dir / f'sample_predictions_{stage_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Sample predictions saved: {save_path}")

    def _save_results_json(self, results: Dict, stage_name: str):
        """Save results to JSON file"""
        save_path = self.results_dir / f'metrics_{stage_name}.json'

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  Metrics saved: {save_path}")





# """
# Enhanced Evaluation Pipeline with Stage-based Assessment
# Supports Stage 0 (pretrained), Stage 1 (100 classes), Stage 2 (incremental)
# """
# import torch
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
# from tqdm import tqdm
# import json
# from datetime import datetime
# from PIL import Image
# import torchvision.transforms as transforms
# from collections import defaultdict
#
#
# class StageEvaluator:
#     """Comprehensive evaluation for different training stages"""
#
#     def __init__(self, model, memory_bank, config, device, results_dir: Path):
#         self.model = model
#         self.memory_bank = memory_bank
#         self.config = config
#         self.device = device
#         self.results_dir = results_dir
#         self.results_dir.mkdir(parents=True, exist_ok=True)
#
#         # Transform for evaluation
#         self.eval_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#
#     @torch.no_grad()
#     def evaluate_stage_0_pretrained(self, train_dataset, test_dataset,
#                                     class_list: List[str],
#                                     stage_name: str = 'stage_0') -> Dict:
#         """
#         Stage 0: Evaluate pretrained backbone (no fine-tuning)
#         Uses only the backbone features + simple classifier
#
#         Args:
#             train_dataset: Training dataset to build prototypes from
#             test_dataset: Test dataset to evaluate on
#             class_list: List of class names
#             stage_name: Name of this stage
#         """
#         print(f"\n{'=' * 80}")
#         print(f"STAGE 0 EVALUATION: Pretrained Backbone Only")
#         print(f"{'=' * 80}")
#
#         self.model.eval()
#
#         # 1. Build prototypes from TRAINING set
#         print("Building prototypes from training features...")
#         train_prototypes = self._build_prototypes_from_data(
#             train_dataset, class_list
#         )
#
#         print(f"Built {len(train_prototypes)} prototypes from training data")
#
#         # 2. Extract features from TEST set
#         print("Extracting test features...")
#         all_embeddings = []
#         all_labels = []
#         all_images = []
#
#         from torch.utils.data import DataLoader
#         test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
#
#         for images, labels in tqdm(test_loader, desc="Extracting test features"):
#             images = images.to(self.device)
#
#             # Get embeddings from model
#             embeddings = self.model(images)
#
#             all_embeddings.append(embeddings.cpu())
#             all_labels.extend(labels.numpy())
#             all_images.extend(images.cpu())
#
#         all_embeddings = torch.cat(all_embeddings, dim=0)
#         all_labels = np.array(all_labels)
#
#         print(f"Extracted features from {len(all_labels)} test samples")
#
#         # 3. Compute predictions
#         predictions, similarities, top5_preds = self._compute_predictions_topk(
#             all_embeddings, train_prototypes, class_list
#         )
#
#         # Convert label indices to class names
#         idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
#         true_label_names = [idx_to_class[label] for label in all_labels]
#
#         # 4. Compute metrics
#         results = self._compute_metrics(
#             np.array(true_label_names), predictions, top5_preds, class_list, stage_name
#         )
#
#         # 5. Visualizations
#         self._save_confusion_matrix(
#             true_label_names, predictions, class_list, stage_name
#         )
#
#         self._save_sample_predictions(
#             all_images[:20], true_label_names[:20],
#             top5_preds[:20], class_list, stage_name
#         )
#
#         # 6. Save results
#         self._save_results_json(results, stage_name)
#
#         return results
#
#     @torch.no_grad()
#     def evaluate_stage_with_memory_bank(self, test_loader, class_list: List[str],
#                                         old_classes: Optional[List[str]] = None,
#                                         new_classes: Optional[List[str]] = None,
#                                         stage_name: str = 'stage_1') -> Dict:
#         """
#         Evaluate using memory bank (for trained stages)
#         Computes metrics for all classes, old classes, and new classes separately
#         """
#         print(f"\n{'=' * 80}")
#         print(f"{stage_name.upper()} EVALUATION: {len(class_list)} classes")
#         if old_classes:
#             print(f"Old classes: {len(old_classes)}, New classes: {len(new_classes)}")
#         print(f"{'=' * 80}")
#
#         self.model.eval()
#
#         # Collect predictions
#         all_embeddings = []
#         all_labels = []
#         all_predictions = []
#         all_similarities = []
#         all_images = []
#
#         idx_to_class = {i: cls for i, cls in enumerate(class_list)}
#
#         for images, labels in tqdm(test_loader, desc="Evaluating"):
#             images = images.to(self.device)
#             embeddings = self.model(images)
#
#             all_embeddings.append(embeddings.cpu())
#
#             for emb, label in zip(embeddings, labels):
#                 # Get prediction from memory bank
#                 result = self.memory_bank.retrieve(
#                     emb,
#                     use_adaptive=self.config.inference.use_adaptive_threshold,
#                     global_threshold=self.config.inference.rejection_threshold,
#                     std_multiplier=self.config.inference.adaptive_threshold_std_mult
#                 )
#
#                 all_predictions.append(result['class_name'])
#                 all_labels.append(idx_to_class[label.item()])
#                 all_similarities.append(result['all_similarities'])
#
#             all_images.extend(images.cpu())
#
#             if len(all_images) >= 100:
#                 break
#
#         all_embeddings = torch.cat(all_embeddings, dim=0)
#
#         # Compute top-k predictions
#         top5_preds = self._compute_topk_from_similarities(
#             all_similarities, k=5
#         )
#
#         # Compute metrics
#         results = self._compute_metrics_with_splits(
#             all_labels, all_predictions, top5_preds,
#             class_list, old_classes, new_classes, stage_name
#         )
#
#         # Visualizations
#         self._save_confusion_matrix(
#             all_labels, all_predictions, class_list, stage_name,
#             max_classes=50  # Limit for readability
#         )
#
#         self._save_sample_predictions(
#             all_images[:20], all_labels[:20],
#             top5_preds[:20], class_list, stage_name
#         )
#
#         # Save results
#         self._save_results_json(results, stage_name)
#
#         return results
#
#     def _build_prototypes_from_data(self, dataset, class_list: List[str]) -> Dict:
#         """Build prototypes from dataset (for stage 0)"""
#         class_embeddings = defaultdict(list)
#
#         print(f"Building prototypes from {len(dataset)} samples...")
#
#         # Use ALL training samples (not just 1000)
#         # Or if too many, use stratified sampling
#         max_samples_per_class = 100
#
#         # Group samples by class
#         class_samples = defaultdict(list)
#         for idx in range(len(dataset)):
#             img, label = dataset[idx]
#             class_name = class_list[label]
#             class_samples[class_name].append(idx)
#
#         # Sample from each class
#         selected_indices = []
#         for class_name, indices in class_samples.items():
#             n_samples = min(max_samples_per_class, len(indices))
#             sampled = np.random.choice(indices, n_samples, replace=False)
#             selected_indices.extend(sampled)
#
#         print(f"Using {len(selected_indices)} samples to build prototypes")
#
#         for idx in tqdm(selected_indices, desc="Building prototypes"):
#             img, label = dataset[idx]
#             img = img.unsqueeze(0).to(self.device)
#
#             with torch.no_grad():
#                 emb = self.model(img).cpu().numpy()[0]
#
#             class_name = class_list[label]
#             class_embeddings[class_name].append(emb)
#
#         # Compute prototypes
#         prototypes = {}
#         for class_name, embeddings in class_embeddings.items():
#             prototype = np.mean(embeddings, axis=0)
#             prototype = prototype / np.linalg.norm(prototype)
#             prototypes[class_name] = torch.tensor(prototype).to(self.device)
#
#         print(f"Built prototypes for {len(prototypes)} classes")
#
#         # Check coverage
#         missing = set(class_list) - set(prototypes.keys())
#         if missing:
#             print(f"WARNING: No prototypes built for {len(missing)} classes: {list(missing)[:5]}")
#
#         return prototypes
#
#     def _compute_predictions_topk(self, embeddings: torch.Tensor,
#                                   prototypes: Dict, class_list: List[str],
#                                   k: int = 5) -> Tuple:
#         """Compute predictions and top-k for given embeddings"""
#         predictions = []
#         similarities_list = []
#         top5_preds = []
#
#         for emb in embeddings:
#             emb = emb.to(self.device)
#
#             # Compute similarities
#             sims = {}
#             for class_name, proto in prototypes.items():
#                 sim = torch.dot(emb, proto).item()
#                 sims[class_name] = sim
#
#             # Top-1
#             best_class = max(sims, key=sims.get)
#             predictions.append(best_class)
#             similarities_list.append(sims)
#
#             # Top-k
#             sorted_classes = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)
#             top5_preds.append(sorted_classes[:k])
#
#         return predictions, similarities_list, top5_preds
#
#     def _compute_topk_from_similarities(self, similarities_list: List[Dict],
#                                         k: int = 5) -> List[List[str]]:
#         """Extract top-k predictions from similarity dictionaries"""
#         topk_preds = []
#
#         for sims in similarities_list:
#             sorted_classes = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)
#             topk_preds.append(sorted_classes[:k])
#
#         return topk_preds
#
#     def _compute_metrics(self, true_labels: np.ndarray,
#                          predictions: List[str],
#                          top5_preds: List[List[str]],
#                          class_list: List[str],
#                          stage_name: str) -> Dict:
#         """Compute comprehensive metrics"""
#         results = {
#             'stage': stage_name,
#             'timestamp': datetime.now().isoformat(),
#             'num_classes': len(class_list),
#             'num_samples': len(true_labels)
#         }
#
#         # Top-1 accuracy
#         correct_top1 = sum([p == l for p, l in zip(predictions, true_labels)])
#         results['top1_accuracy'] = correct_top1 / len(true_labels)
#
#         # Top-5 accuracy
#         correct_top5 = sum([l in preds for l, preds in zip(true_labels, top5_preds)])
#         results['top5_accuracy'] = correct_top5 / len(true_labels)
#
#         # Per-class accuracy
#         per_class_acc = {}
#         for cls in class_list:
#             cls_mask = [l == cls for l in true_labels]
#             if sum(cls_mask) > 0:
#                 cls_correct = sum([p == l for p, l, m in
#                                    zip(predictions, true_labels, cls_mask) if m])
#                 per_class_acc[cls] = cls_correct / sum(cls_mask)
#
#         results['per_class_accuracy'] = per_class_acc
#         results['mean_per_class_accuracy'] = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0
#
#         print(f"\n{stage_name.upper()} Results:")
#         print(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f}")
#         print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
#         print(f"  Mean Per-Class Accuracy: {results['mean_per_class_accuracy']:.4f}")
#
#         return results
#
#     def _compute_metrics_with_splits(self, true_labels: List[str],
#                                      predictions: List[str],
#                                      top5_preds: List[List[str]],
#                                      class_list: List[str],
#                                      old_classes: Optional[List[str]],
#                                      new_classes: Optional[List[str]],
#                                      stage_name: str) -> Dict:
#         """Compute metrics with old/new class splits"""
#         results = self._compute_metrics(
#             np.array(true_labels), predictions, top5_preds, class_list, stage_name
#         )
#
#         # Old class metrics
#         if old_classes:
#             old_mask = [l in old_classes for l in true_labels]
#             if sum(old_mask) > 0:
#                 old_correct_top1 = sum([p == l for p, l, m in
#                                         zip(predictions, true_labels, old_mask) if m])
#                 results['old_classes_top1'] = old_correct_top1 / sum(old_mask)
#
#                 old_correct_top5 = sum([l in preds for l, preds, m in
#                                         zip(true_labels, top5_preds, old_mask) if m])
#                 results['old_classes_top5'] = old_correct_top5 / sum(old_mask)
#             else:
#                 results['old_classes_top1'] = 0.0
#                 results['old_classes_top5'] = 0.0
#
#         # New class metrics
#         if new_classes:
#             new_mask = [l in new_classes for l in true_labels]
#             if sum(new_mask) > 0:
#                 new_correct_top1 = sum([p == l for p, l, m in
#                                         zip(predictions, true_labels, new_mask) if m])
#                 results['new_classes_top1'] = new_correct_top1 / sum(new_mask)
#
#                 new_correct_top5 = sum([l in preds for l, preds, m in
#                                         zip(true_labels, top5_preds, new_mask) if m])
#                 results['new_classes_top5'] = new_correct_top5 / sum(new_mask)
#             else:
#                 results['new_classes_top1'] = 0.0
#                 results['new_classes_top5'] = 0.0
#
#         # Print split metrics if available
#         if 'old_classes_top1' in results and 'new_classes_top1' in results:
#             print(f"  Old Classes Top-1: {results['old_classes_top1']:.4f}")
#             print(f"  New Classes Top-1: {results['new_classes_top1']:.4f}")
#         elif 'old_classes_top1' in results:
#             print(f"  Old Classes Top-1: {results['old_classes_top1']:.4f}")
#         elif 'new_classes_top1' in results:
#             print(f"  New Classes Top-1: {results['new_classes_top1']:.4f}")
#
#         return results
#
#     def _save_confusion_matrix(self, true_labels, predictions,
#                                class_list: List[str], stage_name: str,
#                                max_classes: int = 50):
#         """Save confusion matrix visualization"""
#         # Limit classes for readability
#         if len(class_list) > max_classes:
#             print(f"  Skipping confusion matrix ({len(class_list)} > {max_classes} classes)")
#             return
#
#         cm = confusion_matrix(true_labels, predictions, labels=class_list)
#
#         # Normalize
#         cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
#
#         fig, ax = plt.subplots(figsize=(max(12, len(class_list) // 3),
#                                         max(10, len(class_list) // 3)))
#
#         sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
#                     xticklabels=class_list, yticklabels=class_list,
#                     cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
#
#         ax.set_title(f'Confusion Matrix - {stage_name.upper()}', fontsize=14, pad=20)
#         ax.set_ylabel('True Label', fontsize=12)
#         ax.set_xlabel('Predicted Label', fontsize=12)
#         plt.xticks(rotation=90, fontsize=8)
#         plt.yticks(rotation=0, fontsize=8)
#         plt.tight_layout()
#
#         save_path = self.results_dir / f'confusion_matrix_{stage_name}.png'
#         plt.savefig(save_path, dpi=200, bbox_inches='tight')
#         plt.close()
#
#         print(f"  Confusion matrix saved: {save_path}")
#
#     def _save_sample_predictions(self, images: List[torch.Tensor],
#                                  true_labels: List[str],
#                                  top5_preds: List[List[str]],
#                                  class_list: List[str],
#                                  stage_name: str,
#                                  n_samples: int = 20):
#         """Save sample predictions with top-5 probabilities"""
#         n_samples = min(n_samples, len(images))
#
#         # Create grid
#         n_cols = 4
#         n_rows = (n_samples + n_cols - 1) // n_cols
#
#         fig, axes = plt.subplots(n_rows, n_cols,
#                                  figsize=(4 * n_cols, 4 * n_rows))
#         axes = axes.flatten() if n_samples > 1 else [axes]
#
#         # Denormalize images
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#         for idx in range(n_samples):
#             ax = axes[idx]
#
#             # Denormalize image
#             img = images[idx] * std + mean
#             img = torch.clamp(img, 0, 1)
#             img_np = img.permute(1, 2, 0).numpy()
#
#             ax.imshow(img_np)
#             ax.axis('off')
#
#             # Add predictions
#             true_label = true_labels[idx]
#             top5 = top5_preds[idx]
#
#             title = f"True: {true_label[:20]}\n"
#             title += "Top-5:\n"
#             for i, pred in enumerate(top5, 1):
#                 marker = "✓" if pred == true_label else "✗"
#                 title += f"{i}. {pred[:15]} {marker}\n"
#
#             ax.set_title(title, fontsize=8, ha='left')
#
#         # Hide remaining axes
#         for idx in range(n_samples, len(axes)):
#             axes[idx].axis('off')
#
#         plt.suptitle(f'Sample Predictions - {stage_name.upper()}',
#                      fontsize=14, y=0.995)
#         plt.tight_layout()
#
#         save_path = self.results_dir / f'sample_predictions_{stage_name}.png'
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
#
#         print(f"  Sample predictions saved: {save_path}")
#
#     def _save_results_json(self, results: Dict, stage_name: str):
#         """Save results to JSON file"""
#         save_path = self.results_dir / f'metrics_{stage_name}.json'
#
#         with open(save_path, 'w') as f:
#             json.dump(results, f, indent=2, default=str)
#
#         print(f"  Metrics saved: {save_path}")