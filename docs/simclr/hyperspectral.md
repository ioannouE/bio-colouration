# Hyperspectral Training

Training SimCLR models on hyperspectral data with 408 spectral bands provides the highest spectral resolution for biological analysis, enabling detailed characterization of spectral signatures across organisms.

## Overview

Hyperspectral training leverages the full electromagnetic spectrum from 348-809 nm with 408 equally-spaced bands. This enables analysis of subtle spectral differences that may be crucial for species identification, condition assessment, or evolutionary studies.

## Data Requirements

### File Format
- **Input Files**: `.bil` (Band Interleaved by Line) with `.hdr` headers
- **Spectral Bands**: 408 channels (348-809 nm)
- **Band Spacing**: ~1.13 nm per band
- **Source**: University of St Andrews Lepidoptera Database


## Configuration

### Basic Hyperspectral Config
```yaml
# config_hyperspectral.yaml
data_dir: "data/hyperspectral/lepidoptera"
out_dir: "outputs/hyperspectral"

# Model configuration
backbone: "resnet50"
weights: "DEFAULT"
channels: 408  # Full hyperspectral
input_size: 224

# Training parameters
lr: 0.0005  # Lower LR for high-dimensional data
batch_size: 4   # Reduced for memory constraints
max_epochs: 200
temperature: 0.1

# Hyperspectral specific
hyperspectral_mode: "full"
sample_bands: null  # Use all bands
```

### Memory-Optimized Config
```yaml
# For limited GPU memory
batch_size: 2
```

## Training Commands

### Full Spectrum Training
```bash
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --config configs/config_kornia_hyperspectral.yaml
```

### RGB Approximation
Use bands closest to RGB wavelengths:
```bash
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --config configs/config_kornia_hyperspectral.yaml \
  --rgb-only
```

### Custom Band Sampling
```bash
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --config configs/config_kornia_hyperspectral.yaml \
  --sample-bands 0,50,100,150,200,250,300,350,400
```

### Combined Loss Training
```bash
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --config configs/config_kornia_hyperspectral.yaml \
  --loss-type combined \
  --global-weight 0.6 \
  --local-weight 0.4
```

## Band Sampling Strategies

### Uniform Sampling
```python
# Sample every nth band
def uniform_sampling(total_bands=408, n_samples=50):
    step = total_bands // n_samples
    return list(range(0, total_bands, step))[:n_samples]

bands = uniform_sampling(408, 50)
print(f"Selected bands: {bands}")
```

### Spectral Region Sampling
```python
# Sample from different spectral regions
def spectral_region_sampling():
    regions = {
        'uv': list(range(0, 50)),      # 348-400 nm
        'blue': list(range(50, 120)),   # 400-480 nm  
        'green': list(range(120, 180)), # 480-550 nm
        'red': list(range(180, 250)),   # 550-650 nm
        'nir': list(range(250, 408))    # 650-809 nm
    }
    
    # Sample 10 bands from each region
    selected = []
    for region, bands in regions.items():
        step = len(bands) // 10
        selected.extend(bands[::step][:10])
    
    return sorted(selected)
```

### Principal Component Sampling
```python
# Sample bands based on PCA importance
from sklearn.decomposition import PCA

def pca_band_selection(hyperspectral_data, n_bands=50):
    # Reshape data for PCA
    reshaped = hyperspectral_data.reshape(-1, 408)
    
    # Fit PCA
    pca = PCA(n_components=n_bands)
    pca.fit(reshaped)
    
    # Find most important bands
    loadings = np.abs(pca.components_)
    band_importance = loadings.sum(axis=0)
    selected_bands = np.argsort(band_importance)[-n_bands:]
    
    return sorted(selected_bands)
```

## Data Preprocessing

### Spectral Normalization
```python
def normalize_hyperspectral(data):
    """Normalize hyperspectral data across spectral dimension."""
    # data shape: (batch, 408, height, width)
    
    # Method 1: Per-pixel normalization
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    normalized = (data - mean) / (std + 1e-8)
    
    return normalized

def normalize_per_band(data):
    """Normalize each band independently."""
    normalized = torch.zeros_like(data)
    for i in range(data.shape[1]):
        band = data[:, i]
        normalized[:, i] = (band - band.mean()) / (band.std() + 1e-8)
    
    return normalized
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA

def reduce_spectral_dimensions(data, n_components=50):
    """Reduce spectral dimensions using PCA."""
    original_shape = data.shape
    reshaped = data.reshape(-1, original_shape[1])  # Flatten spatial dims
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)
    
    # Reshape back
    new_shape = (original_shape[0], n_components, original_shape[2], original_shape[3])
    return reduced.reshape(new_shape), pca
```

## Advanced Training Techniques

### Progressive Band Training
```python
# Start with fewer bands, gradually increase
training_schedule = [
    {'epochs': 50, 'bands': 20},
    {'epochs': 100, 'bands': 50},
    {'epochs': 150, 'bands': 100},
    {'epochs': 200, 'bands': 408}
]

for stage in training_schedule:
    # Update model for new band count
    model.update_input_channels(stage['bands'])
    train_model(model, epochs=stage['epochs'])
```

### Spectral Attention
```python
class SpectralAttention(nn.Module):
    def __init__(self, n_bands=408):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_bands, n_bands // 16, 1),
            nn.ReLU(),
            nn.Conv2d(n_bands // 16, n_bands, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights
```

### Memory-Efficient Processing
```python
def process_hyperspectral_chunks(data, model, chunk_size=50):
    """Process hyperspectral data in chunks to save memory."""
    n_bands = data.shape[1]
    embeddings = []
    
    for i in range(0, n_bands, chunk_size):
        chunk = data[:, i:i+chunk_size]
        with torch.cuda.amp.autocast():
            chunk_embedding = model(chunk)
        embeddings.append(chunk_embedding)
    
    # Combine chunk embeddings
    return torch.cat(embeddings, dim=1)
```

## Monitoring and Visualization

### Spectral Analysis During Training
```python
def analyze_spectral_importance(model, data, wavelengths):
    """Analyze which spectral bands are most important."""
    importances = []
    
    for i in range(data.shape[1]):
        # Zero out band i
        masked_data = data.clone()
        masked_data[:, i] = 0
        
        # Compute embedding difference
        original = model(data)
        masked = model(masked_data)
        importance = torch.norm(original - masked, dim=1).mean()
        importances.append(importance.item())
    
    # Plot importance vs wavelength
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, importances)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Importance')
    plt.title('Spectral Band Importance')
    plt.savefig('spectral_importance.png')
```

### Memory Usage Monitoring
```python
def monitor_memory_usage():
    """Monitor GPU memory during hyperspectral training."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

## Troubleshooting

### Memory Issues
```bash
# Error: CUDA out of memory
# Solutions:
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --batch-size 1 \
  --accumulate-grad-batches 16 \
  --precision 16 \
  --sample-bands 0,25,50,75,100,125,150,175,200
```

### Slow Training
```yaml
# Optimization config
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 4

# Model optimization
compile_model: true  # PyTorch 2.0+
```

### Poor Convergence
- Reduce learning rate for high-dimensional data
- Use spectral normalization
- Consider band sampling strategies
- Monitor gradient norms

## Performance Optimization

### Efficient Data Loading
```python
class HyperspectralDataset(Dataset):
    def __init__(self, data_dir, sample_bands=None):
        self.sample_bands = sample_bands
        
    def __getitem__(self, idx):
        # Load only required bands
        data = self.load_bil_file(self.files[idx])
        
        if self.sample_bands:
            data = data[self.sample_bands]
            
        return data
    
    def load_bil_file(self, filepath):
        # Efficient .bil file loading
        with rasterio.open(filepath) as src:
            # Read only required bands
            if self.sample_bands:
                data = src.read(self.sample_bands)
            else:
                data = src.read()
        return torch.from_numpy(data)
```

### Model Optimization
```python
# Use gradient checkpointing for memory efficiency
model = torch.utils.checkpoint.checkpoint_sequential(
    model, segments=4, input=data
)

# Compile model for speed (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

## Evaluation Strategies

### Spectral Signature Analysis
```python
def extract_spectral_signatures(embeddings, labels, wavelengths):
    """Extract representative spectral signatures per class."""
    signatures = {}
    
    for label in np.unique(labels):
        mask = labels == label
        class_embeddings = embeddings[mask]
        
        # Compute mean spectral signature
        mean_signature = class_embeddings.mean(axis=0)
        signatures[label] = mean_signature
    
    return signatures
```

### Cross-Spectral Validation
```python
# Compare different spectral sampling strategies
sampling_strategies = {
    'uniform': uniform_sampling(408, 50),
    'rgb_focused': [100, 150, 200],  # RGB-like bands
    'full_spectrum': list(range(408))
}

results = {}
for name, bands in sampling_strategies.items():
    model = train_hyperspectral_model(selected_bands=bands)
    accuracy = evaluate_model(model)
    results[name] = accuracy
```

## Example Workflows

### Band Selection Experiment
```bash
# Test different band sampling strategies
for n_bands in 10 20 50 100 200 408; do
    python scripts/simclr_birdcolour_kornia_hyperspectral.py \
      --sample-bands $(python utils/generate_uniform_bands.py --n-bands $n_bands) \
      --output-dir outputs/bands_$n_bands
done
```

### Progressive Training
```bash
# Stage 1: Train on sampled bands
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --sample-bands 0,50,100,150,200,250,300,350,400 \
  --epochs 100 \
  --output-dir outputs/stage1

# Stage 2: Fine-tune on full spectrum
python scripts/simclr_birdcolour_kornia_hyperspectral.py \
  --resume outputs/stage1/model.ckpt \
  --epochs 50 \
  --lr 0.0001 \
  --output-dir outputs/stage2
```

## Best Practices

### Data Management
1. Verify .bil/.hdr file pairs
2. Check wavelength calibration
3. Monitor file sizes and loading times
4. Use efficient storage formats

### Training Strategy
1. Start with band sampling for experiments
2. Use progressive training for full spectrum
3. Monitor memory usage carefully
4. Implement gradient checkpointing

### Hyperparameter Guidelines
- Learning rate: 0.0001-0.001 (lower for full spectrum)
- Batch size: 1-8 (depending on GPU memory)
- Temperature: 0.05-0.2 (may need adjustment for high dimensions)
- Epochs: 200-500 (longer for complex spectral patterns)

## Next Steps

- **[Cross-Modal Learning](cross-modal.md)**: Multi-modal fusion strategies
- **[Data Augmentation](augmentation.md)**: Spectral-specific augmentations
- **[Evaluation](../eval/overview.md)**: Analyzing hyperspectral embeddings
