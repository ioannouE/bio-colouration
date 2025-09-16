# Training

This guide covers the training process for SimCLR models in Phenoscape, including configuration, monitoring, and optimization strategies.

## Basic Training Process

### 1. Configuration Setup

Create a configuration file (`config.yaml`) with your training parameters:

```yaml
# Data configuration
data_dir: "path/to/your/data"
out_dir: "outputs/experiment_name"

# Model configuration
backbone: "resnet50"  # resnet18, resnet50, vit_base_patch16_224
weights: "DEFAULT"    # Use pretrained weights or null for random init

# Training parameters
lr: 0.001
batch_size: 32
max_epochs: 100
temperature: 0.1      # Contrastive loss temperature
seed: 42

# Hardware configuration
accelerator: "gpu"    # cpu, gpu, auto
devices: 1
num_workers: 4
deterministic: true

# Logging
log_every_n_steps: 50
```

### 2. Start Training

```bash
cd simclr
python scripts/simclr_birdcolour.py --config config.yaml
```

### 3. Monitor Progress

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir outputs/experiment_name
```

## Configuration Parameters

### Model Architecture

| Parameter | Options | Description |
|-----------|---------|-------------|
| `backbone` | `resnet18`, `resnet50`, `resnet101`, `vit_base_patch16_224` | Encoder architecture |
| `weights` | `DEFAULT`, `null` | Use pretrained weights or random initialization |
| `input_size` | `224`, `256`, `384` | Input image resolution |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lr` | `0.001` | `1e-5` to `1e-1` | Learning rate |
| `batch_size` | `32` | `8` to `512` | Batch size (adjust for GPU memory) |
| `max_epochs` | `100` | `50` to `500` | Maximum training epochs |
| `temperature` | `0.1` | `0.05` to `0.5` | Contrastive loss temperature |
| `T_max` | `max_epochs` | - | Cosine annealing scheduler period |

### Data Augmentation

```yaml
# Augmentation parameters (passed to SimCLRTransform)
transforms:
  input_size: 224
  cj_prob: 0.8        # Color jitter probability
  cj_strength: 0.5    # Color jitter strength
  min_scale: 0.08     # Minimum crop scale
  random_gray_scale: 0.2  # Grayscale probability
  gaussian_blur: 0.5  # Gaussian blur probability
  kernel_size: 0.1    # Blur kernel size
  sigmas: [0.1, 2.0]  # Blur sigma range
```

## Training Strategies

### Learning Rate Scheduling

The framework uses cosine annealing by default:

```yaml
# Cosine annealing scheduler
lr_scheduler: "cosine"
T_max: 100          # Period of cosine annealing
eta_min: 0.0001     # Minimum learning rate
```

### Batch Size Optimization

Choose batch size based on GPU memory:

| GPU Memory | Recommended Batch Size |
|------------|------------------------|
| 8GB | 16-32 |
| 16GB | 32-64 |
| 24GB+ | 64-128 |

### Early Stopping

Monitor validation loss to prevent overfitting:

```yaml
early_stopping:
  monitor: "val_loss"
  patience: 10
  min_delta: 0.001
```

## Advanced Training Options

### Mixed Precision Training

Enable automatic mixed precision for faster training:

```yaml
precision: 16  # Use 16-bit precision
```

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
accumulate_grad_batches: 4  # Effective batch size = batch_size * 4
```

### Multi-GPU Training

Train on multiple GPUs:

```yaml
accelerator: "gpu"
devices: 2  # Use 2 GPUs
strategy: "ddp"  # Distributed data parallel
```

## Monitoring Training

### Key Metrics to Watch

1. **Contrastive Loss**: Should decrease steadily
2. **Learning Rate**: Follow the scheduler curve
3. **GPU Utilization**: Should be >80% for efficiency
4. **Memory Usage**: Monitor for out-of-memory errors

### TensorBoard Visualization

```bash
# Start TensorBoard
tensorboard --logdir outputs/

# View metrics at http://localhost:6006
```

Key plots to monitor:
- Training loss over time
- Learning rate schedule
- Gradient norms
- Sample augmented images

### Training Logs

Check training logs for issues:

```bash
tail -f outputs/experiment_name/logs/training.log
```

## Troubleshooting

### Common Issues

**Out of Memory Errors**
```
RuntimeError: CUDA out of memory
```
Solutions:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Reduce input image size

**Slow Convergence**
- Increase learning rate
- Adjust temperature parameter
- Check data augmentation strength
- Verify data loading efficiency

**Loss Not Decreasing**
- Check data preprocessing
- Verify positive/negative pair generation
- Adjust temperature parameter
- Increase batch size for better negative sampling

### Performance Optimization

**Data Loading**
```yaml
num_workers: 8      # Increase for faster data loading
pin_memory: true    # Speed up GPU transfer
persistent_workers: true  # Keep workers alive
```

**Model Optimization**
```python
# Compile model for PyTorch 2.0+
model = torch.compile(model)
```

## Training Best Practices

### Data Preparation
1. Ensure balanced dataset across categories
2. Remove corrupted or low-quality images
3. Standardize image preprocessing
4. Verify data augmentation quality

### Hyperparameter Tuning
1. Start with default parameters
2. Tune learning rate first
3. Adjust batch size for hardware
4. Fine-tune temperature parameter
5. Optimize augmentation strength

### Experiment Management
1. Use descriptive experiment names
2. Version control configuration files
3. Save model checkpoints regularly
4. Document experiment results

### Validation Strategy
1. Hold out validation set
2. Monitor overfitting
3. Use early stopping
4. Evaluate on downstream tasks

## Example Training Commands

### Basic RGB Training
```bash
python scripts/simclr_birdcolour.py --config configs/rgb_basic.yaml
```

### High-Resolution Training
```bash
python scripts/simclr_birdcolour.py --config configs/rgb_high_res.yaml
```

### Multi-GPU Training
```bash
python scripts/simclr_birdcolour.py --config configs/multi_gpu.yaml
```

### Resume Training
```bash
python scripts/simclr_birdcolour.py \
  --config configs/resume.yaml \
  --resume outputs/experiment/checkpoints/last.ckpt
```

## Output Files

After training, you'll find:

```
outputs/experiment_name/
├── checkpoints/
│   ├── best.ckpt      # Best model checkpoint
│   └── last.ckpt      # Latest checkpoint
├── logs/
│   ├── training.log   # Training logs
│   └── tensorboard/   # TensorBoard logs
└── config.yaml        # Saved configuration
```

## Next Steps

- **[Configuration](configuration.md)**: Detailed parameter explanations
- **[Multispectral Training](multispectral.md)**: Training with UV data
- **[Hyperspectral Training](hyperspectral.md)**: 408-band training
- **[Data Augmentation](augmentation.md)**: Customizing augmentations
