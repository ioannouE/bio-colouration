# RGB Images

Standard RGB (Red, Green, Blue) images are the most common data type supported by Phenoscape. This format is ideal for analyzing visible color patterns in biological organisms.

## Data Format

- **File Types**: `.jpg`, `.jpeg`, `.png`, `.tiff`
- **Channels**: 3 (Red, Green, Blue)

## Data Requirements

### Image Quality
- **Resolution**: Minimum 224x224 pixels (higher resolution preferred).
- **Focus**: Sharp, well-focused images. The algorithm was tested on museum specimens but it should work on natural images as well.
- **Lighting**: Preferably consistent, even illumination.
- **Background**: Clean background or pre-segmented regions of interest.

### Preprocessing Recommendations
- **Segmentation**: Remove background to focus on the organism
- **Standardization**: Consistent orientation and scale (preferably). This can be managed using proper augmentations.
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
data:
  data_dir: "path/to/rgb/data"
output:
  output_dir: "path/to/output/directory"
model:
  name: "simclr"
  backbone: "vit_l_16" # or resnet50
  weights: "ViT_L_16_Weights" # or IMAGENET1K_V2 for resnet50

augmentations:
  input_size: 224
```

## Data Augmentations

RGB images support extensive augmentation options:

- **Geometric**: Rotation, flipping, cropping, perspective changes
- **Color**: Brightness, contrast, saturation, hue adjustments
- **Noise**: Gaussian blur, random erasing
- **Advanced**: Elastic deformation, thin plate spline



## Example Usage

```bash
# Train SimCLR on RGB data
python train/simclr_birdcolour.py --config configs/config_rgb.yaml
```
This will automatically train the model and generate embeddings (csv file) in the output directory.
We also provide the option to generate embeddings from a pre-trained model:

```bash
# Generate embeddings
python embeddings/generate_embeddings.py \
  --model-path outputs/rgb_model.ckpt \
  --data-dir data/rgb_images \
  --output embeddings_rgb.csv
```

## Related Topics

- **[Multispectral Data](multispectral.md)**: Extended spectral information
- **[Training](../simclr/training.md)**: Model training procedures
- **[Examples](../examples/example_rgb.md)**: Practical usage examples
