"""
Enhanced Configuration for Few-Shot Incremental Traffic Sign Recognition
Includes all improvements: better augmentations, adaptive thresholds, etc.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class DataConfig:
    """Dataset configuration with higher resolution support"""
    data_root: Path = Path("prepared_dataset/ready_training")
    img_size: int = 224  # CLIP ViT-B/32 requires 224x224 (FIXED from 384)
    num_workers: int = 4
    initial_classes: int = 100
    incremental_steps: List[int] = field(default_factory=lambda: [50, 50, 100])
    samples_per_class: int = 5

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = "ViT-B/32"  # CLIP model
    pretrained: bool = True
    embedding_dim: int = 512  # INCREASED for better representation
    hidden_dim: int = 1024  # INCREASED projection capacity
    projection_layers: int = 3  # ADDED: deeper projection head
    dropout: float = 0.15  # INCREASED for better regularization
    freeze_backbone_after_base: bool = True
    use_batch_norm: bool = True

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 100
    batch_size: int = 64  # REDUCED for stability with larger images
    samples_per_class_per_batch: int = 4
    learning_rate: float = 5e-5  # REDUCED for fine-tuning
    weight_decay: float = 1e-3  # INCREASED regularization
    temperature: float = 0.05  # REDUCED for sharper similarity
    warmup_epochs: int = 10
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1  # ADDED

@dataclass
class AugmentationConfig:
    """ENHANCED: Stronger, more realistic augmentations"""
    # Basic geometric
    random_resized_crop_scale: tuple = (0.7, 1.0)  # ADJUSTED
    horizontal_flip_prob: float = 0.5

    # ADDED: Rotation for traffic signs
    rotation_degrees: int = 15
    rotation_prob: float = 0.5

    # ENHANCED: Color jitter
    color_jitter_brightness: float = 0.5  # INCREASED
    color_jitter_contrast: float = 0.5    # INCREASED
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1  # ADDED

    # ENHANCED: Blur
    gaussian_blur_prob: float = 0.4  # INCREASED
    gaussian_blur_kernel: int = 23
    gaussian_blur_sigma: tuple = (0.1, 2.5)

    # ADDED: Perspective transform (realistic camera angles)
    perspective_prob: float = 0.3
    perspective_distortion: float = 0.2

    # ADDED: Motion blur (realistic road conditions)
    motion_blur_prob: float = 0.2
    motion_blur_kernel: int = 9

    # ADDED: Noise
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.05

@dataclass
class InferenceConfig:
    """ENHANCED: Per-class adaptive thresholds"""
    # Global threshold (fallback) - LOWERED to reduce unknowns
    rejection_threshold: float = 0.2

    # ENHANCED: Adaptive thresholds - DISABLED for now due to catastrophic forgetting
    use_adaptive_threshold: bool = False
    adaptive_threshold_std_mult: float = 3.0  # INCREASED (was 1.5, too strict)
    adaptive_min_samples: int = 3

    # ADDED: Per-class thresholds - DISABLED for now
    use_per_class_threshold: bool = False
    per_class_percentile: float = 5.0  # LOWERED from 10.0

    # Top-k predictions
    top_k: List[int] = field(default_factory=lambda: [1, 3, 5])

    # ADDED: k-NN search - KEEP ENABLED
    use_knn: bool = True
    knn_k: int = 7  # INCREASED from 5
    knn_weight_by_distance: bool = True

@dataclass
class SystemConfig:
    """Complete system configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # System paths
    checkpoint_dir: Path = Path("./checkpoints")
    embedding_dir: Path = Path("./embeddings")
    results_dir: Path = Path("./results")
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        """Create necessary directories"""
        for path in [self.checkpoint_dir, self.embedding_dir,
                     self.results_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)