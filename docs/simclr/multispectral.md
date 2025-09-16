# Multispectral Training

Training SimCLR models on multispectral data enables analysis of both visible and ultraviolet patterns, providing insights into biological features invisible to human vision.

## Overview

Multispectral training extends standard RGB training to include UV channels, allowing the model to learn representations across a broader electromagnetic spectrum. This is particularly valuable for biological organisms that display UV patterns for communication, camouflage, or mate selection.

## Data Requirements

### File Format
- **Input**: 7-channel TIFF files
- **Channels 1-3**: RGB (visible spectrum)
- **Channels 4-6**: UV spectrum
- **Channel 7**: Binary segmentation mask


## Configuration

### Basic Multispectral Config
```yaml
# config_multispectral.yaml
data_dir: "data/multispectral"
out_dir: "outputs/multispectral"

# Model configuration
backbone: "vit_base_patch16_224"
weights: "DEFAULT"
input_size: 224

# Training parameters
lr: 0.001
batch_size: 16  # Reduced for larger channel count
max_epochs: 100
temperature: 0.1

```

### Channel Configuration Options

| Mode | Channels Used | Description |
|------|---------------|-------------|
| `full` | 1-6 | All visible and UV channels |
| `rgb_only` | 1-3 | Only visible spectrum |
| `uv_only` | 4-6 | Only UV spectrum |

## Training Commands

### Full Spectral Training
```bash
python scripts/simclr_kornia_spectral.py \
  --config configs/config_kornia_multispectral.yaml
```

### RGB-Only Mode
```bash
python scripts/simclr_kornia_spectral.py \
  --config configs/config_kornia_multispectral.yaml \
  --rgb-only
```

### USML Color Space
Train using tetrahedral color space for UV-sensitive analysis:
```bash
python scripts/simclr_kornia_spectral.py \
  --config configs/config_kornia_multispectral.yaml \
  --usml
```

## Advanced Training Options

### Loss Function Types

#### Global Loss (Standard)
```bash
python scripts/simclr_kornia_spectral.py \
  --loss-type global
```

#### Local Patch Loss
```bash
python scripts/simclr_kornia_spectral.py \
  --loss-type local
```

#### Combined Loss
```bash
python scripts/simclr_kornia_spectral.py \
  --loss-type combined \
  --global-weight 0.7 \
  --local-weight 0.3
```

### Model Checkpoints
Use pretrained multispectral classifier as backbone:
```bash
python scripts/simclr_kornia_spectral.py \
  --model-checkpoint pretrained/multispectral_classifier.ckpt
```

## Data Preprocessing

### Channel Normalization
```python
# Normalize each channel independently
def normalize_multispectral(data):
    """Normalize 6-channel multispectral data."""
    normalized = torch.zeros_like(data)
    for i in range(6):
        channel = data[:, i]
        normalized[:, i] = (channel - channel.mean()) / channel.std()
    return normalized
```

### Mask Application
```python
# Apply segmentation mask to all channels
def apply_mask(data, mask):
    """Apply binary mask to multispectral data."""
    # Expand mask to all channels
    mask_expanded = mask.unsqueeze(1).repeat(1, 6, 1, 1)
    return data * mask_expanded
```

## Augmentation Strategies

### Spectral-Aware Augmentations
```yaml
# Multispectral augmentation config
augmentations:
  # Geometric (applied to all channels)
  random_crop: 0.8
  random_flip: 0.5
  random_rotation: 15
  
  # Color (channel-specific)
  rgb_color_jitter: 0.4
  uv_intensity_jitter: 0.2
  
  # Spectral
  channel_dropout: 0.1  # Randomly drop channels
  spectral_shift: 0.05  # Shift spectral response
```

### Custom Augmentation Pipeline
```python
import kornia.augmentation as K

# Multispectral augmentation pipeline
multispectral_transforms = K.AugmentationSequential(
    K.RandomResizedCrop(224, scale=(0.2, 1.0)),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.2),
    K.RandomRotation(degrees=15),
    K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),  # Applied to RGB channels
    same_on_batch=False
)
```

## Monitoring Training

### Key Metrics for Multispectral Data

1. **Channel-wise Loss**: Monitor loss per spectral band
2. **UV vs RGB Separation**: Ensure both modalities contribute
3. **Mask Utilization**: Verify segmentation mask effectiveness
4. **Memory Usage**: 6-7 channels require more GPU memory

### Visualization During Training
```bash
# Visualize augmentations
python scripts/simclr_kornia_spectral.py \
  --visualize \
  --config configs/config_kornia_multispectral.yaml
```

## Troubleshooting

### Common Issues

**Memory Errors with 7 Channels**
```yaml
# Reduce batch size
batch_size: 8
# Use gradient accumulation
accumulate_grad_batches: 4
```

**Poor UV Channel Learning**
- Check UV channel data quality
- Adjust channel-specific normalization
- Verify mask application
- Consider UV-specific augmentations

**Channel Imbalance**
```python
# Weight channels differently in loss
channel_weights = torch.tensor([1.0, 1.0, 1.0, 1.5, 1.5, 1.5])  # Higher UV weights
```

### Performance Optimization

**Memory Efficient Training**
```yaml
# Mixed precision
precision: 16

# Efficient data loading
num_workers: 4
pin_memory: true
prefetch_factor: 2
```

**Channel-wise Processing**
```python
# Process channels in groups to save memory
def process_channels_grouped(data, group_size=3):
    results = []
    for i in range(0, data.shape[1], group_size):
        group = data[:, i:i+group_size]
        processed = model_forward(group)
        results.append(processed)
    return torch.cat(results, dim=1)
```

## Evaluation Strategies

### Cross-Modal Validation
```python
# Evaluate RGB vs UV representations separately
rgb_embeddings = generate_embeddings(model, rgb_data)
uv_embeddings = generate_embeddings(model, uv_data)

# Compare clustering quality
from sklearn.metrics import silhouette_score
rgb_score = silhouette_score(rgb_embeddings, labels)
uv_score = silhouette_score(uv_embeddings, labels)
```

### Spectral Analysis
```python
# Analyze which channels contribute most
def analyze_channel_importance(model, data):
    importances = []
    for i in range(6):
        # Mask out channel i
        masked_data = data.clone()
        masked_data[:, i] = 0
        
        # Compute embedding difference
        original_emb = model(data)
        masked_emb = model(masked_data)
        importance = torch.norm(original_emb - masked_emb, dim=1).mean()
        importances.append(importance.item())
    
    return importances
```

## Best Practices

### Data Quality
1. Verify channel registration accuracy
2. Check for spectral artifacts
3. Validate segmentation masks
4. Ensure consistent exposure across channels

### Training Strategy
1. Start with RGB-only training for baseline
2. Gradually introduce UV channels
3. Monitor channel-specific contributions
4. Use appropriate loss weighting

### Hyperparameter Tuning
1. Adjust batch size for memory constraints
2. Fine-tune temperature for multi-modal data
3. Optimize augmentation strength per channel
4. Consider channel-specific learning rates

## Example Workflows

### Comparative Analysis
```bash
# Train RGB-only model
python scripts/simclr_kornia_spectral.py --rgb-only --config config_rgb.yaml

# Train UV-only model  
python scripts/simclr_kornia_spectral.py --uv-only --config config_uv.yaml

# Train full multispectral model
python scripts/simclr_kornia_spectral.py --config config_full.yaml

# Compare results
python eval_vis/compare_modalities.py \
  --rgb-embeddings rgb_embeddings.csv \
  --uv-embeddings uv_embeddings.csv \
  --full-embeddings full_embeddings.csv
```

### Progressive Training
```bash
# Stage 1: RGB pretraining
python scripts/simclr_kornia_spectral.py --rgb-only --epochs 50

# Stage 2: Add UV channels
python scripts/simclr_kornia_spectral.py \
  --resume rgb_model.ckpt \
  --full-spectral \
  --lr 0.0001 \
  --epochs 50
```

## Next Steps

- **[Hyperspectral Training](hyperspectral.md)**: 408-band training
- **[Cross-Modal Learning](cross-modal.md)**: Multi-modal fusion strategies
- **[Data Augmentation](augmentation.md)**: Spectral-specific augmentations
