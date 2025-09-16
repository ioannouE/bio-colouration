# Data Augmentation

Data augmentation is crucial for SimCLR training, creating diverse views of the same image to learn robust representations. Phenoscape uses the Kornia library for GPU-accelerated augmentations across RGB, multispectral, and hyperspectral data.

## Overview

SimCLR requires two augmented views of each image to create positive pairs for contrastive learning. The quality and diversity of augmentations directly impact the learned representations' robustness and biological relevance.

## Kornia-Based Augmentations

### Standard RGB Augmentations

```python
import kornia.augmentation as K

# Basic RGB augmentation pipeline
rgb_transforms = K.AugmentationSequential(
    K.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.2),
    K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
    same_on_batch=False
)
```

### Advanced Augmentations

```python
# Extended augmentation set
advanced_transforms = K.AugmentationSequential(
    K.RandomResizedCrop(224, scale=(0.2, 1.0)),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.3),
    K.RandomRotation(degrees=15, p=0.7),
    K.RandomPerspective(distortion_scale=0.2, p=0.4),
    K.RandomThinPlateSpline(scale=0.2, p=0.3),
    K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
    K.RandomErase(scale=(0.02, 0.33), ratio=(0.3, 3.3), p=0.5),
    K.RandomPosterize(bits=3, p=0.4),
    K.RandomSharpness(sharpness=0.5, p=0.4),
    same_on_batch=False
)
```

## Augmentation Categories

### Geometric Transformations

| Augmentation | Parameters | Biological Relevance |
|--------------|------------|---------------------|
| `RandomResizedCrop` | `scale=(0.08, 1.0)` | Simulates different viewing distances |
| `RandomHorizontalFlip` | `p=0.5` | Natural left-right symmetry |
| `RandomVerticalFlip` | `p=0.2` | Less common but valid orientation |
| `RandomRotation` | `degrees=15` | Natural pose variations |
| `RandomPerspective` | `distortion_scale=0.2` | Camera angle variations |

### Color Transformations

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| `ColorJitter` | `brightness=0.4, contrast=0.4` | Lighting condition variations |
| `RandomGrayscale` | `p=0.2` | Forces learning of shape features |
| `RandomPosterize` | `bits=3` | Reduces color depth |
| `RandomSharpness` | `sharpness=0.5` | Image quality variations |

### Noise and Distortions

| Augmentation | Parameters | Effect |
|--------------|------------|--------|
| `RandomGaussianBlur` | `kernel_size=(3,3), sigma=(0.1,2.0)` | Motion blur, focus variations |
| `RandomErase` | `scale=(0.02,0.33)` | Occlusion simulation |
| `RandomThinPlateSpline` | `scale=0.2` | Elastic deformations |

## Multispectral Augmentations

### Channel-Specific Augmentations

```python
class MultispectralAugmentation:
    def __init__(self):
        # RGB channels (1-3)
        self.rgb_transforms = K.AugmentationSequential(
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)
        )
        
        # UV channels (4-6)
        self.uv_transforms = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0, std=0.1, p=0.3),
            K.RandomGamma((0.8, 1.2), p=0.5),
            K.RandomBrightness(brightness=0.2, p=0.4)
        )
        
        # Geometric (applied to all channels)
        self.geometric_transforms = K.AugmentationSequential(
            K.RandomResizedCrop(224, scale=(0.2, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=10)
        )
    
    def __call__(self, multispectral_data):
        # Apply geometric transforms to all channels
        data = self.geometric_transforms(multispectral_data)
        
        # Split channels
        rgb_channels = data[:, :3]
        uv_channels = data[:, 3:6]
        mask_channel = data[:, 6:7] if data.shape[1] == 7 else None
        
        # Apply modality-specific transforms
        rgb_augmented = self.rgb_transforms(rgb_channels)
        uv_augmented = self.uv_transforms(uv_channels)
        
        # Recombine channels
        if mask_channel is not None:
            return torch.cat([rgb_augmented, uv_augmented, mask_channel], dim=1)
        else:
            return torch.cat([rgb_augmented, uv_augmented], dim=1)
```

### Spectral-Specific Augmentations

```python
# UV-specific augmentations
uv_specific_transforms = K.AugmentationSequential(
    K.RandomGaussianNoise(mean=0, std=0.05, p=0.4),  # UV sensor noise
    K.RandomGamma((0.7, 1.3), p=0.6),                # UV sensitivity variations
    K.RandomContrast(contrast=(0.8, 1.2), p=0.5),    # UV contrast adjustments
)

# Spectral channel dropout
class SpectralChannelDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            # Randomly zero out one spectral channel
            channel_to_drop = torch.randint(0, x.shape[1], (1,))
            x[:, channel_to_drop] = 0
        return x
```

## Hyperspectral Augmentations

### Band-Aware Augmentations

```python
class HyperspectralAugmentation:
    def __init__(self, n_bands=408):
        self.n_bands = n_bands
        
        # Geometric transforms (applied to all bands)
        self.geometric = K.AugmentationSequential(
            K.RandomResizedCrop(224, scale=(0.3, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=5)  # Smaller rotation for hyperspectral
        )
        
        # Spectral transforms
        self.spectral_noise = K.RandomGaussianNoise(mean=0, std=0.02, p=0.3)
        self.spectral_gamma = K.RandomGamma((0.9, 1.1), p=0.4)
    
    def __call__(self, hyperspectral_data):
        # Apply geometric transforms
        data = self.geometric(hyperspectral_data)
        
        # Apply spectral augmentations
        data = self.spectral_noise(data)
        data = self.spectral_gamma(data)
        
        # Spectral band shuffling (occasionally)
        if torch.rand(1) < 0.1:
            data = self.spectral_band_shuffle(data)
        
        return data
    
    def spectral_band_shuffle(self, data, max_shift=5):
        """Slightly shuffle spectral bands to simulate calibration variations."""
        shifts = torch.randint(-max_shift, max_shift + 1, (data.shape[1],))
        shuffled_data = torch.zeros_like(data)
        
        for i, shift in enumerate(shifts):
            new_idx = torch.clamp(i + shift, 0, data.shape[1] - 1)
            shuffled_data[:, i] = data[:, new_idx]
        
        return shuffled_data
```

### Spectral Region Augmentations

```python
def spectral_region_augmentation(data, wavelengths):
    """Apply different augmentations to different spectral regions."""
    # Define spectral regions
    uv_mask = wavelengths < 400
    visible_mask = (wavelengths >= 400) & (wavelengths < 700)
    nir_mask = wavelengths >= 700
    
    # UV region: add noise (sensor limitations)
    if uv_mask.any():
        uv_noise = torch.randn_like(data[:, uv_mask]) * 0.05
        data[:, uv_mask] += uv_noise
    
    # Visible region: color jittering equivalent
    if visible_mask.any():
        visible_data = data[:, visible_mask]
        # Apply brightness/contrast variations
        brightness_factor = torch.rand(1) * 0.4 + 0.8  # 0.8-1.2
        contrast_factor = torch.rand(1) * 0.4 + 0.8
        data[:, visible_mask] = visible_data * contrast_factor + brightness_factor
    
    # NIR region: different noise characteristics
    if nir_mask.any():
        nir_noise = torch.randn_like(data[:, nir_mask]) * 0.03
        data[:, nir_mask] += nir_noise
    
    return data
```

## Configuration

### YAML Configuration

```yaml
# Augmentation configuration
augmentations:
  # Geometric
  random_crop:
    size: 224
    scale: [0.08, 1.0]
    ratio: [0.75, 1.33]
  
  random_flip:
    horizontal: 0.5
    vertical: 0.2
  
  random_rotation:
    degrees: 15
    probability: 0.7
  
  # Color
  color_jitter:
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4
    hue: 0.1
    probability: 0.8
  
  random_grayscale:
    probability: 0.2
  
  # Blur and noise
  gaussian_blur:
    kernel_size: [3, 3]
    sigma: [0.1, 2.0]
    probability: 0.5
  
  # Advanced
  random_erase:
    scale: [0.02, 0.33]
    ratio: [0.3, 3.3]
    probability: 0.5
```

### Programmatic Configuration

```python
def create_augmentation_pipeline(config):
    """Create augmentation pipeline from configuration."""
    transforms = []
    
    # Geometric transforms
    if 'random_crop' in config:
        crop_config = config['random_crop']
        transforms.append(
            K.RandomResizedCrop(
                size=crop_config['size'],
                scale=crop_config['scale'],
                ratio=crop_config['ratio']
            )
        )
    
    if 'random_flip' in config:
        flip_config = config['random_flip']
        transforms.append(K.RandomHorizontalFlip(p=flip_config['horizontal']))
        transforms.append(K.RandomVerticalFlip(p=flip_config['vertical']))
    
    # Color transforms
    if 'color_jitter' in config:
        cj_config = config['color_jitter']
        transforms.append(
            K.ColorJitter(
                brightness=cj_config['brightness'],
                contrast=cj_config['contrast'],
                saturation=cj_config['saturation'],
                hue=cj_config['hue'],
                p=cj_config['probability']
            )
        )
    
    return K.AugmentationSequential(*transforms, same_on_batch=False)
```

## Augmentation Strategies

### Progressive Augmentation

```python
class ProgressiveAugmentation:
    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs
        self.base_transforms = self._create_base_transforms()
        self.advanced_transforms = self._create_advanced_transforms()
    
    def get_transforms(self, current_epoch):
        """Get augmentation pipeline based on training progress."""
        progress = current_epoch / self.total_epochs
        
        if progress < 0.3:
            # Early training: gentle augmentations
            return self.base_transforms
        elif progress < 0.7:
            # Mid training: mixed augmentations
            return self._create_mixed_transforms(progress)
        else:
            # Late training: full augmentations
            return self.advanced_transforms
```

### Adaptive Augmentation

```python
class AdaptiveAugmentation:
    def __init__(self, initial_strength=0.5):
        self.strength = initial_strength
        self.performance_history = []
    
    def update_strength(self, current_loss, target_loss=1.0):
        """Adjust augmentation strength based on training progress."""
        if current_loss > target_loss:
            # Training struggling, reduce augmentation
            self.strength = max(0.1, self.strength * 0.95)
        else:
            # Training progressing well, increase augmentation
            self.strength = min(1.0, self.strength * 1.05)
    
    def get_transforms(self):
        """Get augmentation pipeline with current strength."""
        return K.AugmentationSequential(
            K.RandomResizedCrop(224, scale=(0.08, 1.0)),
            K.ColorJitter(
                brightness=0.4 * self.strength,
                contrast=0.4 * self.strength,
                saturation=0.4 * self.strength,
                hue=0.1 * self.strength,
                p=0.8
            ),
            K.RandomGaussianBlur(
                (3, 3), 
                (0.1, 2.0 * self.strength), 
                p=0.5 * self.strength
            )
        )
```

## Visualization and Debugging

### Augmentation Visualization

```python
def visualize_augmentations(image, transforms, n_samples=8):
    """Visualize different augmented versions of an image."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_samples):
        augmented = transforms(image.unsqueeze(0))[0]
        
        # Convert to displayable format
        if augmented.shape[0] == 3:  # RGB
            display_img = augmented.permute(1, 2, 0)
        else:  # Multispectral - show RGB channels
            display_img = augmented[:3].permute(1, 2, 0)
        
        axes[i].imshow(display_img.clamp(0, 1))
        axes[i].set_title(f'Augmentation {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=300)
```

### Augmentation Analysis

```python
def analyze_augmentation_impact(model, dataloader, transforms):
    """Analyze how augmentations affect model predictions."""
    model.eval()
    original_embeddings = []
    augmented_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Original embeddings
            original = model(batch)
            original_embeddings.append(original)
            
            # Augmented embeddings
            augmented = model(transforms(batch))
            augmented_embeddings.append(augmented)
    
    # Compute similarity between original and augmented
    original_embeddings = torch.cat(original_embeddings)
    augmented_embeddings = torch.cat(augmented_embeddings)
    
    similarities = F.cosine_similarity(original_embeddings, augmented_embeddings)
    
    return {
        'mean_similarity': similarities.mean().item(),
        'std_similarity': similarities.std().item(),
        'min_similarity': similarities.min().item(),
        'max_similarity': similarities.max().item()
    }
```

## Best Practices

### Augmentation Design Principles

1. **Biological Relevance**: Ensure augmentations reflect natural variations
2. **Modality Awareness**: Use appropriate augmentations for each spectral range
3. **Gradual Introduction**: Start with simple augmentations, add complexity
4. **Balance**: Avoid over-augmentation that destroys biological information

### Parameter Guidelines

| Data Type | Crop Scale | Color Jitter | Rotation | Flip Probability |
|-----------|------------|--------------|----------|------------------|
| RGB | (0.08, 1.0) | 0.4 | ±15° | 0.5 |
| Multispectral | (0.2, 1.0) | 0.3 | ±10° | 0.5 |
| Hyperspectral | (0.3, 1.0) | 0.2 | ±5° | 0.3 |

### Common Pitfalls

1. **Over-augmentation**: Too strong augmentations can hurt performance
2. **Modality Mismatch**: Using RGB augmentations on UV data
3. **Geometric Distortions**: Excessive perspective changes for biological data
4. **Channel Inconsistency**: Different augmentations across spectral channels

## Troubleshooting

### Performance Issues

**Augmentations Too Weak**
- Increase augmentation probabilities
- Add more transformation types
- Increase parameter ranges

**Augmentations Too Strong**
- Reduce transformation magnitudes
- Lower augmentation probabilities
- Remove complex transformations

**Memory Issues**
- Use `same_on_batch=True` for identical augmentations
- Reduce batch size
- Simplify augmentation pipeline

### Quality Assessment

```python
def assess_augmentation_quality(original_images, augmented_images):
    """Assess if augmentations preserve important features."""
    # Compute structural similarity
    ssim_scores = []
    for orig, aug in zip(original_images, augmented_images):
        ssim = structural_similarity(orig.numpy(), aug.numpy(), channel_axis=0)
        ssim_scores.append(ssim)
    
    return {
        'mean_ssim': np.mean(ssim_scores),
        'std_ssim': np.std(ssim_scores),
        'quality_assessment': 'good' if np.mean(ssim_scores) > 0.5 else 'poor'
    }
```

## Next Steps

- **[Training](training.md)**: Integrating augmentations into training pipeline
- **[Cross-Modal Learning](cross-modal.md)**: Modality-specific augmentation strategies
- **[Evaluation](../eval/overview.md)**: Assessing augmentation impact on embeddings
