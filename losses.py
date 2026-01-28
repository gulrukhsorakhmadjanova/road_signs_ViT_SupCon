"""
Supervised Contrastive Loss for Few-Shot Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss from:
    "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Optimized for few-shot learning with small batch sizes.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, embedding_dim] L2-normalized embeddings
            labels: [batch_size] class labels

        Returns:
            loss: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out diagonal (self-similarity)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(dim=1)
        # Avoid division by zero
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss