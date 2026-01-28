"""
Enhanced Data Augmentations for Traffic Sign Recognition
Includes realistic transforms: rotation, perspective, motion blur, etc.
"""
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageFilter

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class MotionBlur:
    """Apply motion blur to simulate camera/vehicle movement"""

    def __init__(self, kernel_size=9, angle_range=(-45, 45)):
        self.kernel_size = kernel_size
        self.angle_range = angle_range

    def __call__(self, img):
        angle = random.uniform(*self.angle_range)
        kernel = np.zeros((self.kernel_size, self.kernel_size))

        # Create motion blur kernel
        center = self.kernel_size // 2
        for i in range(self.kernel_size):
            offset = int((i - center) * np.tan(np.radians(angle)))
            if 0 <= center + offset < self.kernel_size:
                kernel[i, center + offset] = 1

        kernel = kernel / kernel.sum()

        # Apply kernel
        img_array = np.array(img)
        from scipy.ndimage import convolve
        blurred = np.zeros_like(img_array)

        for channel in range(img_array.shape[2]):
            blurred[:, :, channel] = convolve(
                img_array[:, :, channel], kernel, mode='reflect'
            )

        return Image.fromarray(blurred.astype(np.uint8))


class GaussianNoise:
    """Add Gaussian noise to simulate sensor noise"""

    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0, 1)


def get_train_transforms(config):
    """
    ENHANCED: Training transforms with realistic augmentations
    """
    img_size = config.data.img_size
    aug_config = config.augmentation

    transforms_list = [
        # 1. Random resized crop
        transforms.RandomResizedCrop(
            img_size,
            scale=aug_config.random_resized_crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),

        # 2. Random horizontal flip
        transforms.RandomHorizontalFlip(p=aug_config.horizontal_flip_prob),

        # 3. ADDED: Random rotation (traffic signs can be at angles)
        transforms.RandomApply([
            transforms.RandomRotation(
                degrees=aug_config.rotation_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ], p=aug_config.rotation_prob),

        # 4. ADDED: Perspective transform (realistic camera angles)
        transforms.RandomApply([
            transforms.RandomPerspective(
                distortion_scale=aug_config.perspective_distortion,
                p=1.0,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ], p=aug_config.perspective_prob),

        # 5. ENHANCED: Color jitter
        transforms.ColorJitter(
            brightness=aug_config.color_jitter_brightness,
            contrast=aug_config.color_jitter_contrast,
            saturation=aug_config.color_jitter_saturation,
            hue=aug_config.color_jitter_hue
        ),

        # 6. ENHANCED: Gaussian blur
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=aug_config.gaussian_blur_kernel,
                sigma=aug_config.gaussian_blur_sigma
            )
        ], p=aug_config.gaussian_blur_prob),

        # 7. Convert to tensor
        transforms.ToTensor(),

        # 8. ADDED: Gaussian noise (after tensor conversion)
        transforms.RandomApply([
            GaussianNoise(std=aug_config.gaussian_noise_std)
        ], p=aug_config.gaussian_noise_prob),

        # 9. Normalize with CLIP values
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ]

    return transforms.Compose(transforms_list)


def get_val_transforms(config):
    """
    Validation/test transforms (no augmentation)
    Uses higher resolution for better accuracy
    """
    img_size = config.data.img_size

    return transforms.Compose([
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])


def get_test_time_augmentation_transforms(config, n_augments=5):
    """
    ADDED: Test-time augmentation for ensemble predictions
    """
    img_size = config.data.img_size

    # Base transform
    base_transforms = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ]

    # Augmented transforms
    augmented_transforms = []
    for _ in range(n_augments):
        augmented_transforms.append(
            transforms.Compose([
                transforms.Resize(
                    int(img_size * random.uniform(0.9, 1.1)),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            ])
        )

    return [transforms.Compose(base_transforms)] + augmented_transforms