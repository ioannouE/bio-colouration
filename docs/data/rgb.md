# RGB Images

Standard RGB (Red, Green, Blue) images are the most common data type supported by Phenoscape. This format is ideal for analyzing visible color patterns in biological organisms.

## Data Format

- **File Types**: `.jpg`, `.jpeg`, `.png`, `.tiff`
- **Channels**: 3 (Red, Green, Blue)
- **Bit Depth**: 8-bit or 16-bit per channel
- **Color Space**: sRGB recommended

## Data Requirements

### Image Quality
- **Resolution**: Minimum 224x224 pixels (higher resolution preferred)
- **Focus**: Sharp, well-focused images
- **Lighting**: Consistent, even illumination
- **Background**: Clean background or pre-segmented regions of interest

### Preprocessing Recommendations
- **Segmentation**: Remove background to focus on the organism
- **Standardization**: Consistent orientation and scale
- **Quality Control**: Remove blurry, overexposed, or underexposed images

## Directory Structure

Organize your RGB images in the following structure:

```
data/
├── species_sex_001.jpg
├── species_sex_002.jpg
├── ...
└── metadata.csv
```


## Configuration

For RGB data, use these settings in your `config.yaml`:

```yaml
data_dir: "path/to/rgb/data"
input_size: 224
channels: 3
backbone: "resnet50"  # or "vit_base_patch16_224"
weights: "DEFAULT"
```

## Data Augmentation

RGB images support extensive augmentation options:

- **Geometric**: Rotation, flipping, cropping, perspective changes
- **Color**: Brightness, contrast, saturation, hue adjustments
- **Noise**: Gaussian blur, random erasing
- **Advanced**: Elastic deformation, thin plate spline

## Best Practices

### Image Acquisition
1. Use consistent lighting conditions
2. Maintain uniform distance and angle
3. Ensure high contrast between subject and background
4. Capture multiple views when possible

### Data Preparation
1. Standardize image sizes and orientations
2. Apply consistent preprocessing steps
3. Remove low-quality images
4. Balance dataset across categories

### Training Considerations
- Start with pretrained weights for better convergence
- Use appropriate batch sizes based on GPU memory
- Monitor for overfitting with validation data
- Adjust learning rate based on dataset size

## Example Usage

```bash
# Train SimCLR on RGB data
python scripts/simclr_birdcolour.py --config config_rgb.yaml

# Generate embeddings
python scripts/generate_embeddings.py \
  --model-path outputs/rgb_model.ckpt \
  --data-dir data/rgb_images \
  --output embeddings_rgb.csv
```

## Related Topics

- **[Multispectral Data](multispectral.md)**: Extended spectral information
- **[Training](../simclr/training.md)**: Model training procedures
- **[Examples](../examples/example_rgb.md)**: Practical usage examples
