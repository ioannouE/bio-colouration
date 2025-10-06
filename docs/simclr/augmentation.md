# Data Augmentation

Data augmentation is crucial for PhenoCLR training, creating diverse views of the same image to learn robust representations. Phenoscape uses the Kornia library for augmentations across RGB, multispectral, and hyperspectral data.

## Overview

PhenoCLR requires two augmented views of each image to create positive pairs for contrastive learning. The quality and diversity of augmentations directly impact the learned representations' robustness and biological relevance.


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


## Best Practices

### Augmentation Design Principles

1. **Biological Relevance**: Ensure augmentations reflect natural variations
2. **Modality Awareness**: Use appropriate augmentations for each spectral range
3. **Gradual Introduction**: Start with simple augmentations, add complexity
4. **Balance**: Avoid over-augmentation that destroys biological information



## Next Steps

- **[Training](training.md)**: Integrating augmentations into training pipeline
- **[Cross-Modal Learning](cross-modal.md)**: Modality-specific augmentation strategies
- **[Evaluation](../eval/overview.md)**: Assessing augmentation impact on embeddings
