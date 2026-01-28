"""
Inference module for few-shot incremental recognition.

Performs nearest-prototype classification with open-set rejection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Inference engine for prototype-based classification.

    Features:
    - Nearest-prototype classification
    - Open-set rejection
    - Top-k predictions
    - Confidence scores
    - Batch inference
    """

    def __init__(
            self,
            model: nn.Module,
            memory_bank,
            config,
            device: str = 'cuda',
            rejection_threshold: float = 0.5
    ):
        self.model = model.to(device).eval()
        self.memory_bank = memory_bank
        self.config = config
        self.device = device
        self.rejection_threshold = rejection_threshold

        # Per-class thresholds (optional)
        self.per_class_thresholds = None

        logger.info(
            f"InferenceEngine initialized: "
            f"threshold={rejection_threshold:.3f}, "
            f"{memory_bank.num_classes} classes"
        )

    def set_rejection_threshold(self, threshold: float):
        """Set global rejection threshold."""
        self.rejection_threshold = threshold
        logger.info(f"Rejection threshold set to {threshold:.3f}")

    def set_per_class_thresholds(self, thresholds: Dict[int, float]):
        """Set per-class rejection thresholds."""
        self.per_class_thresholds = thresholds
        logger.info(f"Per-class thresholds set for {len(thresholds)} classes")

    @torch.no_grad()
    def predict(
            self,
            images: torch.Tensor,
            return_embeddings: bool = False,
            return_confidences: bool = True,
            top_k: int = 1
    ) -> Dict:
        """
        Predict classes for input images.

        Args:
            images: [batch_size, 3, H, W] input images
            return_embeddings: Whether to return embeddings
            return_confidences: Whether to return confidence scores
            top_k: Return top-k predictions

        Returns:
            Dictionary with predictions and optional outputs
        """
        self.model.eval()
        images = images.to(self.device)

        # Extract embeddings
        embeddings = self.model(images, normalize=True)

        # Get predictions from memory bank
        predictions, distances = self.memory_bank.predict(
            embeddings,
            return_distances=True,
            top_k=top_k
        )

        # Convert distances to similarities (for cosine distance)
        similarities = 1 - distances

        # Apply rejection threshold
        top_similarities = similarities[:, 0]  # Top-1 similarities

        # Determine which predictions to reject
        if self.per_class_thresholds is not None:
            # Use per-class thresholds
            rejected = torch.zeros(len(predictions), dtype=torch.bool)
            for i, pred in enumerate(predictions[:, 0]):
                class_threshold = self.per_class_thresholds.get(
                    pred.item(),
                    self.rejection_threshold
                )
                rejected[i] = top_similarities[i] < class_threshold
        else:
            # Use global threshold
            rejected = top_similarities < self.rejection_threshold

        # Mark rejected predictions as -1 (unknown)
        final_predictions = predictions.clone()
        final_predictions[rejected] = -1

        # Build result dictionary
        result = {
            'predictions': final_predictions,  # [batch_size, top_k]
            'rejected': rejected  # [batch_size]
        }

        if return_confidences:
            result['confidences'] = similarities  # [batch_size, top_k]
            result['top_confidence'] = top_similarities  # [batch_size]

        if return_embeddings:
            result['embeddings'] = embeddings

        return result

    @torch.no_grad()
    def predict_batch(
            self,
            dataloader,
            return_paths: bool = False
    ) -> Dict:
        """
        Predict for entire dataset.

        Args:
            dataloader: DataLoader with images
            return_paths: Whether to include image paths

        Returns:
            Dictionary with all predictions
        """
        all_predictions = []
        all_confidences = []
        all_rejected = []
        all_labels = []
        all_paths = []

        for batch in dataloader:
            if len(batch) == 3:
                images, labels, paths = batch
                if return_paths:
                    all_paths.extend(paths)
            else:
                images, labels = batch

            # Predict
            result = self.predict(
                images,
                return_confidences=True,
                top_k=self.config.inference.top_k
            )

            all_predictions.append(result['predictions'].cpu())
            all_confidences.append(result['confidences'].cpu())
            all_rejected.append(result['rejected'].cpu())
            all_labels.append(labels)

        # Concatenate results
        output = {
            'predictions': torch.cat(all_predictions, dim=0),
            'confidences': torch.cat(all_confidences, dim=0),
            'rejected': torch.cat(all_rejected, dim=0),
            'labels': torch.cat(all_labels, dim=0)
        }

        if return_paths:
            output['paths'] = all_paths

        return output

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        if class_idx == -1:
            return "UNKNOWN"
        return self.memory_bank.class_metadata[class_idx]['name']

    def explain_prediction(
            self,
            image: torch.Tensor,
            top_k: int = 5
    ) -> Dict:
        """
        Provide detailed explanation for a single prediction.

        Args:
            image: [3, H, W] or [1, 3, H, W] input image
            top_k: Number of top classes to explain

        Returns:
            Dictionary with prediction details
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Get prediction
        result = self.predict(
            image,
            return_embeddings=True,
            return_confidences=True,
            top_k=top_k
        )

        # Extract information
        predictions = result['predictions'][0]  # [top_k]
        confidences = result['confidences'][0]  # [top_k]
        embedding = result['embeddings'][0]  # [embedding_dim]
        rejected = result['rejected'][0].item()

        # Build explanation
        explanation = {
            'predicted_class': self.get_class_name(predictions[0].item()),
            'predicted_idx': predictions[0].item(),
            'confidence': confidences[0].item(),
            'rejected': rejected,
            'rejection_threshold': self.rejection_threshold,
            'top_k_classes': []
        }

        for i in range(min(top_k, len(predictions))):
            class_idx = predictions[i].item()
            if class_idx == -1:
                continue

            explanation['top_k_classes'].append({
                'rank': i + 1,
                'class_name': self.get_class_name(class_idx),
                'class_idx': class_idx,
                'confidence': confidences[i].item(),
                'similarity': confidences[i].item()
            })

        # Add nearest prototype information
        if not rejected and predictions[0].item() != -1:
            prototype = self.memory_bank.get_prototype(predictions[0].item())
            distance_to_prototype = torch.norm(
                embedding - prototype, p=2
            ).item()
            explanation['distance_to_prototype'] = distance_to_prototype

        return explanation


class OpenSetDetector:
    """
    Specialized detector for open-set recognition.

    Focuses on identifying unknown/out-of-distribution samples.
    """

    def __init__(
            self,
            inference_engine: InferenceEngine,
            method: str = 'threshold'
    ):
        self.inference_engine = inference_engine
        self.method = method

        logger.info(f"OpenSetDetector initialized: method={method}")

    @torch.no_grad()
    def detect_unknowns(
            self,
            images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect which images are unknown/out-of-distribution.

        Args:
            images: [batch_size, 3, H, W]

        Returns:
            is_unknown: [batch_size] boolean tensor
            scores: [batch_size] anomaly scores
        """
        result = self.inference_engine.predict(
            images,
            return_confidences=True,
            top_k=1
        )

        confidences = result['top_confidence']

        if self.method == 'threshold':
            # Simple threshold-based detection
            is_unknown = result['rejected']
            scores = 1 - confidences  # Higher score = more anomalous

        elif self.method == 'entropy':
            # Use prediction entropy
            # Get top-k confidences
            result_topk = self.inference_engine.predict(
                images,
                return_confidences=True,
                top_k=min(10, self.inference_engine.memory_bank.num_classes)
            )
            confidences_topk = result_topk['confidences']

            # Normalize to probabilities
            probs = F.softmax(confidences_topk, dim=1)

            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

            # Higher entropy = more uncertain = more likely unknown
            scores = entropy

            # Threshold on entropy
            threshold = 1.0  # Tune this
            is_unknown = entropy > threshold

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return is_unknown, scores

    def evaluate_open_set(
            self,
            known_images: torch.Tensor,
            unknown_images: torch.Tensor
    ) -> Dict:
        """
        Evaluate open-set detection performance.

        Args:
            known_images: Images from known classes
            unknown_images: Images from unknown classes

        Returns:
            Metrics dictionary
        """
        # Detect on known samples
        known_unknown, known_scores = self.detect_unknowns(known_images)

        # Detect on unknown samples
        unknown_unknown, unknown_scores = self.detect_unknowns(unknown_images)

        # Compute metrics
        # True positives: correctly identified unknowns
        tp = unknown_unknown.sum().item()
        # False positives: known classified as unknown
        fp = known_unknown.sum().item()
        # True negatives: known classified as known
        tn = (~known_unknown).sum().item()
        # False negatives: unknown classified as known
        fn = (~unknown_unknown).sum().item()

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

        return metrics