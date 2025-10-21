# Multispectral Data

Multispectral images extend beyond visible light to capture ultraviolet (UV) information, providing insights into color patterns invisible to the human eye but crucial for many biological organisms.

## Data Format

- **File Type**: `.tif` (TIFF format)
- **Channels**: 7 total channels
  - Channels 1-3: Visible spectrum (Red, Green, Blue)
  - Channels 4-6: Ultraviolet (UV) spectrum
  - Channel 7: Binary segmentation mask
- **Spatial Resolution**: Consistent across all channels

## Channel Configuration
Note: This is the configuration used for our multispectral data. Other configurations should also work but may need adjusting the data processing pipeline (`data/multispectral_data.py`).

| Channel | Wavelength Range | Description |
|---------|------------------|-------------|
| 1 | ~620-750 nm | Red (visible) |
| 2 | ~495-570 nm | Green (visible) |
| 3 | ~450-495 nm | Blue (visible) |
| 4 | ~315-400 nm | UV-A |
| 5 | ~280-315 nm | UV-B |
| 6 | ~100-280 nm | UV-C |
| 7 | Binary | Segmentation mask |

## Data Requirements

### Image Quality
- **Registration**: All channels must be spatially aligned

### Segmentation Mask
The 7th channel should contain a binary mask where:
- `255` (white): Region of interest (organism)
- `0` (black): Background to be ignored

## Directory Structure

```
data/multispectral/
│   ├── species_sex_001.tif
│   ├── species_sex_002.tif
│   └── ... 
└── metadata.csv
```

## Configuration

For multispectral data training (`configs/config_multispectral.yaml`):

```yaml
data:
  data_dir: "path/to/multispectral/data"
output:
  out_dir: "path/to/output/directory"

model:
  name: 'simclr'  
  backbone: 'vit_l_16'
  weights: 'ViT_L_16_Weights'

criterion:
    type: 'crossmodal' # 'crossmodal', 'local', 'global', 'none'
    global_weight: 1.0
    local_weight: 1.0

augmentations:
  input_size: 224
```

## Training Options

### RGB-Only Mode
Use only visible channels (1-3):

```bash
python train/simclr_kornia_spectral_multimodal.py --rgb-only
```

### Full Spectral Mode
Use all 6 spectral channels:

```bash
python train/simclr_kornia_spectral_multimodal.py --config configs/config_multispectral.yaml
```

### Tetrahedral Color Space
Convert to tetrahedral color space for UV-sensitive analysis:

```bash
python train/simclr_kornia_spectral_multimodal.py --usml
```

### UV-Only Mode
Use only UV channels (4-6):

```bash
python train/simclr_kornia_spectral_multimodal.py --uv-only
```

### Cross-Modal Learning
Compare RGB and UV representations:

```bash
python train/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type separate \
  --use-modality-specific
```

## Fusion Strategies

### Concatenation
Combine RGB and UV channels:
```yaml
fusion_type: "concat"
```

### Separate Encoders
Use different encoders for RGB and UV:
```yaml
fusion_type: "separate"
use_modality_specific: true
```

### Attention Fusion
Use attention mechanisms to combine modalities:
```yaml
fusion_type: "attention"
```


## Example Usage

```bash
# Train on full multispectral data
python train/simclr_kornia_spectral_multimodal.py \
  --config configs/config_kornia_multispectral.yaml

# Train with cross-modal contrastive learning
python train/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type attention \
  --global-weight 0.7 \
  --local-weight 0.3
```

## Related Topics

- **[RGB Images](rgb.md)**: Standard visible spectrum data
- **[Hyperspectral Data](hyperspectral.md)**: High-resolution spectral data
- **[Cross-Modal Learning](../simclr/cross-modal.md)**: Multi-modal training strategies
