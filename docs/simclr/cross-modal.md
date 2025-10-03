# Cross-Modal Learning

Cross-modal contrastive learning enables comparison between different spectral modalities (RGB vs UV) within the same framework, providing insights into how different parts of the electromagnetic spectrum encode biological information.

## Overview

Traditional SimCLR contrasts different augmented views of the same image. Cross-modal learning extends this by contrasting RGB channels against UV channels, allowing the model to learn relationships between visible and ultraviolet patterns in biological organisms.

## Conceptual Framework

### Standard SimCLR
```
Image → [Augmentation 1, Augmentation 2] → Contrastive Loss
```

### Cross-Modal SimCLR
```
Multispectral Image → [RGB Channels, UV Channels] → Cross-Modal Contrastive Loss
```

## Implementation Approaches

### 1. Separate Modality Encoders

Use different encoders for RGB and UV channels:

```python
class CrossModalSimCLR(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.rgb_encoder = create_encoder(backbone, input_channels=3)
        self.uv_encoder = create_encoder(backbone, input_channels=3)
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, rgb_data, uv_data):
        rgb_features = self.rgb_encoder(rgb_data)
        uv_features = self.uv_encoder(uv_data)
        
        rgb_projected = self.projection_head(rgb_features)
        uv_projected = self.projection_head(uv_features)
        
        return rgb_projected, uv_projected
```

### 2. Shared Encoder with Modality Tokens

```python
class SharedModalityEncoder(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.encoder = create_encoder(backbone, input_channels=6)
        self.modality_tokens = nn.Parameter(torch.randn(2, 128))  # RGB, UV tokens
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
    
    def forward(self, multispectral_data):
        # Split into RGB and UV
        rgb_data = multispectral_data[:, :3]
        uv_data = multispectral_data[:, 3:6]
        
        # Process with shared encoder
        features = self.encoder(multispectral_data)
        
        # Apply modality-specific attention
        rgb_features = self.attention(features, self.modality_tokens[0:1])
        uv_features = self.attention(features, self.modality_tokens[1:2])
        
        return rgb_features, uv_features
```

## Training Configuration

### Basic Cross-Modal Config
```yaml
# config_cross_modal.yaml
data_dir: "data/multispectral"
out_dir: "outputs/cross_modal"

# Model configuration
backbone: "vit_l_base_patch16"
fusion_type: "separate"  # "concat", "separate", "attention"
use_modality_specific: true

# Cross-modal parameters
cross_modal_weight: 1.0
intra_modal_weight: 0.5
temperature: 0.1

# Training parameters
lr: 0.001
batch_size: 16
max_epochs: 100
```

### Fusion Strategy Options

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `concat` | Concatenate RGB and UV channels | Simple fusion |
| `separate` | Separate encoders for each modality | Maximum flexibility |
| `attention` | Attention-based fusion | Adaptive weighting |

## Training Commands

### Basic Cross-Modal Training
```bash
python train/simclr_birdcolour_kornia_spectral_multimodal.py \
  --config configs/config_cross_modal.yaml \
  --fusion-type separate
```

### Attention-Based Fusion
```bash
python train/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type attention \
  --use-modality-specific
```

### Weighted Loss Training
```bash
python train/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type separate \
  --cross-modal-weight 0.8 \
  --intra-modal-weight 0.2
```

## Example Workflows

### Comparative Study
```bash
# Train RGB-only baseline
python train/simclr_kornia_spectral.py --rgb-only

# Train UV-only baseline  
python train/simclr_kornia_spectral.py --uv-only

# Train cross-modal model
python train/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type separate



## Next Steps

- **[Data Augmentation](augmentation.md)**: Modality-specific augmentation strategies
- **[Evaluation](../eval/overview.md)**: Cross-modal evaluation techniques
- **[Examples](../examples/example_multispectral.md)**: Practical cross-modal applications
