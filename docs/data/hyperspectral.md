# Hyperspectral Data

Hyperspectral imaging provides the highest spectral resolution, capturing 408 spectral bands across the electromagnetic spectrum. This data type is particularly valuable for detailed spectral analysis of biological organisms.

## Data Format

- **File Types**: `.bil` (Band Interleaved by Line) with `.hdr` header files
- **Channels**: 408 spectral bands
- **Wavelength Range**: 348-809 nm (equally distributed)
- **Bit Depth**: 16-bit per band
- **Source**: University of St Andrews Lepidoptera Hyperspectral Database

## Data Structure

Each hyperspectral image consists of two files:

### .bil File
Contains the raw spectral data in band-interleaved format:
```
specimen_001.bil  # Raw hyperspectral data
```

### .hdr File
Contains metadata and wavelength information:
```
specimen_001.bil.hdr  # Header with wavelength calibration
```

## Wavelength Information

The 408 bands are equally distributed between 348-809 nm:
- **Band spacing**: ~1.13 nm per band
- **Spectral range**: UV to near-infrared
- **Exact wavelengths**: Specified in each `.hdr` file

## Directory Structure

```
data/hyperspectral/lepidoptera/
├── species_1/
│   ├── specimen_001.bil
│   ├── specimen_001.bil.hdr
│   ├── specimen_002.bil
│   ├── specimen_002.bil.hdr
│   └── ...
├── species_2/
│   ├── specimen_001.bil
│   ├── specimen_001.bil.hdr
│   └── ...
└── metadata.csv
```

## Configuration

For hyperspectral data training:

```yaml
data_dir: "data/hyperspectral/lepidoptera"
input_size: 224
channels: 408  # Full hyperspectral
backbone: "resnet50"
hyperspectral_mode: "full"
sample_bands: null  # Use all bands, or specify subset
```

## Band Sampling

### Full Spectrum
Use all 408 bands:
```bash
python train/simclr_birdcolour_kornia_hyperspectral.py
```

### RGB Approximation
Use bands closest to RGB wavelengths (450nm, 550nm, 650nm):
```bash
python train/simclr_birdcolour_kornia_hyperspectral.py --rgb-only
```

### Custom Band Selection
Sample specific bands:
```bash
python train/simclr_birdcolour_kornia_hyperspectral.py \
  --sample-bands 0,50,100,150,200,250,300,350,400
```

## Data Reading

The framework uses a Python reimplementation of the MATLAB `read_iml_hyp.m` function:

```python
from scripts.read_iml_hyp import read_hyperspectral_image

# Read hyperspectral data
data, wavelengths = read_hyperspectral_image('specimen_001.bil')
print(f"Data shape: {data.shape}")  # (height, width, 408)
print(f"Wavelengths: {wavelengths[:5]}...")  # First 5 wavelengths
```

## RGB Conversion

For visualization, convert to RGB using specific wavelengths:

```python
# Wavelengths closest to RGB
rgb_bands = {
    'red': 650,    # nm
    'green': 550,  # nm  
    'blue': 450    # nm
}

# Find closest band indices
rgb_indices = find_closest_wavelengths(wavelengths, rgb_bands)
rgb_image = data[:, :, rgb_indices]
```

## Training Options

### Loss Functions
Choose between different loss strategies:

```bash
# Global loss only (standard SimCLR)
python train/simclr_birdcolour_kornia_hyperspectral.py --loss-type global

# Local patch-based loss
python train/simclr_birdcolour_kornia_hyperspectral.py --loss-type local

# Combined global and local loss
python train/simclr_birdcolour_kornia_hyperspectral.py \
  --loss-type combined \
  --global-weight 0.7 \
  --local-weight 0.3
```

### Memory Optimization
For large hyperspectral data:

```yaml
batch_size: 8  # Reduce for 408-channel data
num_workers: 4
pin_memory: true
```

## Data Preprocessing

### Spectral Normalization
```python
# Normalize across spectral dimension
normalized_data = (data - data.mean(axis=2, keepdims=True)) / \
                  data.std(axis=2, keepdims=True)
```

### Dimensionality Reduction
For computational efficiency:
```python
# PCA on spectral dimension
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
reduced_data = pca.fit_transform(data.reshape(-1, 408))
```

## Best Practices

### Data Handling
1. Verify `.bil` and `.hdr` file pairs
2. Check wavelength calibration consistency
3. Handle missing or corrupted bands
4. Monitor memory usage with full spectrum

### Training Considerations
- Start with band sampling for initial experiments
- Use gradient checkpointing for memory efficiency
- Consider spectral-aware augmentations
- Monitor for spectral overfitting

### Computational Requirements
- **Memory**: 16GB+ RAM recommended for full spectrum
- **GPU**: 8GB+ VRAM for reasonable batch sizes
- **Storage**: Hyperspectral files are large (>100MB each)

## Example Usage

```bash
# Train on full hyperspectral data
python train/simclr_birdcolour_kornia_hyperspectral.py \
  --config configs/config_kornia_hyperspectral.yaml

# Train on sampled bands
python train/simclr_birdcolour_kornia_hyperspectral.py \
  --sample-bands 0,25,50,75,100,125,150,175,200 \
  --loss-type combined

# Generate embeddings only (skip training)
python train/simclr_birdcolour_kornia_hyperspectral.py \
  --test-embeddings-only \
  --model-checkpoint outputs/hyperspectral_model.ckpt
```

## Troubleshooting

### Common Issues
- **File format errors**: Verify `.bil`/`.hdr` file integrity
- **Memory errors**: Reduce batch size or use band sampling
- **Wavelength misalignment**: Check header file consistency
- **Slow training**: Consider dimensionality reduction

### Performance Tips
- Use band sampling for initial experiments
- Enable mixed precision training
- Use data loading optimizations
- Consider spectral binning for efficiency

## Related Topics

- **[Multispectral Data](multispectral.md)**: Lower-resolution spectral data
- **[Training](../simclr/training.md)**: General training procedures
