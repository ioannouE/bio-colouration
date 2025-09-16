# SimCLR Configuration

This guide covers configuration options for SimCLR training in the Phenoscape framework. Learn how to set up training parameters, data paths, model architectures, and optimization settings.

## Overview

SimCLR configuration is managed through YAML files that define all aspects of training including model architecture, data loading, augmentations, and optimization parameters. The framework supports RGB, multispectral, and hyperspectral data configurations.

## Configuration Structure

### Basic Configuration Template

```yaml
# Basic SimCLR configuration template
experiment_name: "simclr_rgb_baseline"
output_dir: "experiments/simclr_rgb"
seed: 42

# Model configuration
model:
  backbone: "resnet50"
  projection_dim: 128
  projection_hidden_dim: 2048
  temperature: 0.07
  
# Data configuration
data:
  type: "rgb"  # rgb, multispectral, hyperspectral
  data_dir: "data/rgb/"
  batch_size: 256
  num_workers: 8
  image_size: 224
  
# Training configuration
training:
  epochs: 100
  learning_rate: 0.0003
  weight_decay: 1e-4
  warmup_epochs: 10
  
# Augmentation configuration
augmentations:
  random_crop:
    scale: [0.08, 1.0]
    ratio: [0.75, 1.33]
  color_jitter:
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4
    hue: 0.1
    probability: 0.8
  random_grayscale:
    probability: 0.2
  gaussian_blur:
    kernel_size: [3, 3]
    sigma: [0.1, 2.0]
    probability: 0.5
```

## Model Configuration

### Backbone Networks

```yaml
model:
  # ResNet backbones
  backbone: "resnet50"     # Standard choice

  # Vision Transformer
  backbone: "vit_l_16"    # Transformer-based
```



## Data Configuration

### RGB Data Configuration

```yaml
data:
  type: "rgb"
  data_dir: "data/rgb/"
  split_file: "data/splits.json"  # Optional train/val/test splits
  
  # Image preprocessing
  image_size: 224
  normalize: true
  mean: [0.485, 0.456, 0.406]  # ImageNet means
  std: [0.229, 0.224, 0.225]   # ImageNet stds
  
  # Data loading
  batch_size: 256
  num_workers: 8
  pin_memory: true
  drop_last: true
```

### Multispectral Data Configuration

```yaml
data:
  type: "multispectral"
  data_dir: "data/multispectral/"
  n_channels: 7  # RGB + UV + mask
  
  # Channel configuration
  rgb_channels: [0, 1, 2]      # RGB channel indices
  uv_channels: [3, 4, 5]       # UV channel indices
  mask_channel: 6              # Segmentation mask
  
  # Preprocessing
  image_size: 224
  normalize_per_channel: true
  clip_values: [0.0, 1.0]      # Value clipping range
  
  # Data loading
  batch_size: 128              # Smaller batch for memory
  num_workers: 4
```

### Hyperspectral Data Configuration

```yaml
data:
  type: "hyperspectral"
  data_dir: "data/hyperspectral/"
  n_bands: 408                 # Number of spectral bands
  
  # Band sampling strategy
  band_sampling: "random_50"   # Sample 50 random bands
  # band_sampling: "uniform_50"  # Sample 50 uniform bands
  # band_sampling: "important_50" # Sample 50 most important bands
  # band_sampling: "all"         # Use all bands (memory intensive)
  
  # Wavelength range
  wavelength_min: 400          # Minimum wavelength (nm)
  wavelength_max: 800          # Maximum wavelength (nm)
  
  # Preprocessing
  image_size: 224
  normalize_bands: true
  remove_bad_bands: true       # Remove noisy bands
  
  # Data loading
  batch_size: 64               # Small batch for memory constraints
  num_workers: 2
```

## Training Configuration

### Optimization Settings

```yaml
training:
  epochs: 100
  
  # Optimizer configuration
  optimizer: "adamw"           # adam, adamw, sgd
  learning_rate: 0.0003
  weight_decay: 1e-4
  momentum: 0.9                # For SGD
  betas: [0.9, 0.999]         # For Adam/AdamW
  
  # Learning rate scheduling
  lr_scheduler: "cosine"       # cosine, step, exponential
  warmup_epochs: 10
  min_lr: 1e-6                # Minimum learning rate
  
  # Step scheduler settings (if using step)
  step_size: 30
  gamma: 0.1
```

### Advanced Training Options

```yaml
training:
  # Mixed precision training
  use_amp: true                # Automatic mixed precision
  
  # Gradient settings
  max_grad_norm: 1.0          # Gradient clipping
  accumulate_grad_batches: 1   # Gradient accumulation
  
  # Checkpointing
  save_every_n_epochs: 10
  save_top_k: 3               # Keep top 3 checkpoints
  monitor_metric: "train_loss" # Metric to monitor
  
  # Early stopping
  patience: 20                # Epochs without improvement
  min_delta: 0.001           # Minimum change threshold
```

## Augmentation Configuration

### Standard RGB Augmentations

```yaml
augmentations:
  # Geometric transforms
  random_resized_crop:
    size: 224
    scale: [0.08, 1.0]
    ratio: [0.75, 1.33]
    
  random_horizontal_flip:
    probability: 0.5
    
  random_vertical_flip:
    probability: 0.2
    
  random_rotation:
    degrees: 15
    probability: 0.7
    
  # Color transforms
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
    
  random_erase:
    scale: [0.02, 0.33]
    ratio: [0.3, 3.3]
    probability: 0.5
```

### Multispectral-Specific Augmentations

```yaml
augmentations:
  # Apply different augmentations to different channel groups
  channel_specific:
    rgb_channels:
      color_jitter:
        brightness: 0.4
        contrast: 0.4
        saturation: 0.4
        hue: 0.1
        probability: 0.8
      random_grayscale:
        probability: 0.2
        
    uv_channels:
      gaussian_noise:
        mean: 0.0
        std: 0.05
        probability: 0.3
      gamma_correction:
        gamma_range: [0.8, 1.2]
        probability: 0.5
        
  # Spectral channel dropout
  spectral_dropout:
    probability: 0.1
    max_channels: 1
```

### Hyperspectral Augmentations

```yaml
augmentations:
  # Spectral-aware augmentations
  spectral_noise:
    noise_type: "gaussian"     # gaussian, uniform, salt_pepper
    intensity: 0.02
    probability: 0.3
    
  spectral_shift:
    max_shift: 5               # Maximum band shift
    probability: 0.1
    
  band_dropout:
    max_bands: 10              # Maximum bands to drop
    probability: 0.2
    
  # Reduced geometric augmentations for hyperspectral
  random_resized_crop:
    size: 224
    scale: [0.3, 1.0]          # Less aggressive cropping
    
  random_rotation:
    degrees: 5                 # Smaller rotation
    probability: 0.5
```

## Hardware Configuration

### GPU Settings

```yaml
hardware:
  # GPU configuration
  gpus: [0, 1, 2, 3]          # GPU indices to use
  distributed: true            # Use distributed training
  sync_batchnorm: true        # Synchronize batch norm across GPUs
  
  # Memory optimization
  gradient_checkpointing: true # Trade compute for memory
  pin_memory: true
  non_blocking: true
```

### Memory Management

```yaml
hardware:
  # Batch size scaling
  auto_scale_batch_size: true  # Automatically find optimal batch size
  max_batch_size: 512         # Maximum batch size to try
  
  # Memory efficient settings
  dataloader_num_workers: 4    # Reduce if memory constrained
  prefetch_factor: 2          # Reduce prefetching
  
  # For large models/data
  cpu_offload: true           # Offload to CPU when possible
  activation_checkpointing: true
```

## Logging and Monitoring

### Experiment Tracking

```yaml
logging:
  # Experiment tracking
  use_wandb: true
  wandb_project: "phenoscape-simclr"
  wandb_entity: "your-username"
  
  # TensorBoard logging
  use_tensorboard: true
  tensorboard_dir: "logs/tensorboard"
  
  # Logging frequency
  log_every_n_steps: 50
  val_check_interval: 1.0     # Validate every epoch
```

### Metrics to Track

```yaml
logging:
  metrics:
    - "train_loss"
    - "val_loss"
    - "learning_rate"
    - "epoch_time"
    - "gpu_memory"
    
  # Additional metrics for evaluation
  eval_metrics:
    - "silhouette_score"
    - "adjusted_rand_index"
    - "embedding_variance"
```

## Environment-Specific Configurations

### Local Development

```yaml
# config/local.yaml
data:
  batch_size: 32              # Smaller batch size
  num_workers: 2              # Fewer workers
  
training:
  epochs: 10                  # Quick testing
  
hardware:
  gpus: [0]                   # Single GPU
  distributed: false
```

### HPC Cluster

```yaml
# config/hpc.yaml
data:
  batch_size: 512             # Large batch size
  num_workers: 16             # Many workers
  
training:
  epochs: 200                 # Full training
  use_amp: true              # Mixed precision
  
hardware:
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]  # Multi-GPU
  distributed: true
  
slurm:
  partition: "gpu"
  time: "48:00:00"
  mem: "128G"
  cpus_per_task: 16
```

## Configuration Validation

### Required Fields

```yaml
# Minimum required configuration
experiment_name: "required"
model:
  backbone: "required"
  projection_dim: "required"
data:
  type: "required"
  data_dir: "required"
training:
  epochs: "required"
  learning_rate: "required"
```

### Configuration Inheritance

```yaml
# Base configuration
defaults:
  - base_config
  - model: resnet50
  - data: rgb
  - augmentations: standard
  
# Override specific settings
model:
  projection_dim: 256         # Override base setting

training:
  learning_rate: 0.001        # Override base setting
```

## Example Configurations

### Quick Start RGB

```yaml
experiment_name: "rgb_quickstart"
model:
  backbone: "resnet18"
  projection_dim: 128
data:
  type: "rgb"
  data_dir: "data/rgb/"
  batch_size: 64
training:
  epochs: 50
  learning_rate: 0.001
```

### Production Multispectral

```yaml
experiment_name: "multispectral_production"
model:
  backbone: "resnet50"
  projection_dim: 256
  temperature: 0.07
data:
  type: "multispectral"
  data_dir: "data/multispectral/"
  batch_size: 128
  n_channels: 7
training:
  epochs: 200
  learning_rate: 0.0003
  use_amp: true
hardware:
  gpus: [0, 1, 2, 3]
  distributed: true
```

### Memory-Efficient Hyperspectral

```yaml
experiment_name: "hyperspectral_efficient"
model:
  backbone: "resnet18"        # Lighter model
  projection_dim: 128
data:
  type: "hyperspectral"
  data_dir: "data/hyperspectral/"
  batch_size: 32             # Small batch
  band_sampling: "random_50" # Sample bands
training:
  epochs: 100
  gradient_checkpointing: true
hardware:
  gpus: [0]
  cpu_offload: true
```

## Configuration Best Practices

### Performance Optimization

1. **Batch Size**: Start with largest batch size that fits in memory
2. **Workers**: Set `num_workers = 4 * num_gpus` as starting point
3. **Mixed Precision**: Use `use_amp: true` for faster training
4. **Pin Memory**: Enable for GPU training

### Memory Management

1. **Gradient Checkpointing**: Enable for large models
2. **Band Sampling**: Use for hyperspectral data
3. **Smaller Batches**: Reduce if out of memory
4. **CPU Offload**: For very large datasets

### Reproducibility

1. **Set Seeds**: Always set `seed` for reproducibility
2. **Deterministic**: Enable deterministic operations
3. **Version Control**: Track configuration files
4. **Environment**: Document Python/CUDA versions

## Next Steps

- **[Training Guide](training.md)**: Learn how to run training with your configuration
- **[Multispectral Training](multispectral.md)**: Specific multispectral configurations
- **[Hyperspectral Training](hyperspectral.md)**: Hyperspectral-specific settings
