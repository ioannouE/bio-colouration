# Quick Start

Get up and running with Phenoscape in just a few steps.

## Basic RGB Analysis

### 1. Prepare Your Data

Organize your RGB images in a directory structure:

```
data/
├── image1.jpg
├── image2.jpg
└── ...
```

For the evaluation purposes, the images' names should be in the format `species_sex_001.jpg`, `species_sex_002.jpg`, etc. For training SimCLR and generating embeddings, the images' names could be in any format e.g., `image1.jpg`, `image2.jpg`, etc.


### 2. Configure Training

Create or modify `config.yaml`:

```yaml
data_dir: "path/to/your/data"
out_dir: "path/to/your/output"
backbone: "resnet50"
weights: "DEFAULT"
lr: 0.001
batch_size: 32
max_epochs: 100
```

### 3. Train SimCLR Model

```bash
cd simclr
python scripts/simclr_birdcolour.py --config config.yaml
```

### 4. Generate Embeddings

After training, generate embeddings for analysis:

```bash
python scripts/generate_embeddings.py --model-path outputs/model.ckpt --data-dir path/to/data
```

### 5. Visualize Results

```bash
cd ../eval_vis
python evaluate.py \
  --csv_path embeddings.csv \
  --image_dir path/to/images \
  --output_dir visualizations \
  --labels species color \
  --emb_num 1024
```

## Multispectral Analysis

For 7-channel multispectral data:

```bash
cd simclr
python scripts/simclr_kornia_spectral.py --config configs/config_kornia_multispectral.yaml
```

## Hyperspectral Analysis

For 408-band hyperspectral data:

```bash
cd simclr
python scripts/simclr_birdcolour_kornia_hyperspectral.py --config configs/config_kornia_hyperspectral.yaml
```

## Expected Outputs

After running the pipeline, you'll get:

- **Trained model**: `checkpoints/model.ckpt`
- **Embeddings**: CSV file with feature vectors
- **Visualizations**: UMAP/t-SNE plots with image thumbnails
- **Metrics**: Statistical analysis results

## Next Steps

- **[Data Types](../data/rgb.md)**: Learn about different data formats
- **[SimCLR Configuration](../simclr/configuration.md)**: Customize training parameters
- **[Evaluation Tools](../eval/overview.md)**: Explore analysis options
- **[Examples](../examples/example_rgb.md)**: See detailed use cases


