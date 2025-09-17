# Overview

**Phenoscape** is a comprehensive framework for analyzing organismal colour patterns using artificial intelligence. This project focuses on understanding colour pattern variation within and between biological species through modern machine learning techniques.

## What is Phenoscape?

Phenoscape combines state-of-the-art computer vision and contrastive learning methods to analyze biological colour patterns across multiple spectral modalities. The framework is built around SimCLR (Simple Contrastive Learning of Visual Representations) and provides tools for both training models and evaluating embeddings.

## Key Capabilities

### Multi-Modal Data Support
- **RGB Images**: Standard colour images of biological organisms
- **Multispectral Data**: 7-channel data including visible and ultraviolet spectrum
- **Hyperspectral Data**: 408-band spectral imaging from specialised databases

### Advanced Analysis Tools
- **Contrastive Learning**: Generate meaningful embeddings using SimCLR
- **Cross-Modal Learning**: Compare RGB and UV channel representations
- **Statistical Analysis**: Comprehensive evaluation metrics and visualization tools
- **Embedding Visualisation**: Interactive plots with image thumbnails

## Research Applications

Phenoscape is designed for researchers in:

- **Evolutionary Biology**: Understanding adaptive colouration patterns
- **Computer Vision**: Developing robust visual representations for biological data
- **Comparative Morphology**: Analyzing phenotypic variation across species
- **Spectral Imaging**: Working with multispectral and hyperspectral datasets


## Project Structure

The framework consists of two main modules:

1. **SimCLR Module** (`simclr/`): Core contrastive learning implementation
2. **Evaluation & Visualization** (`eval_vis/`): Analysis and visualization tools

## Next Steps

- **[Installation](installation.md)**: Set up your environment
- **[Quick Start](quickstart.md)**: Run your first analysis
- **[Data Types](../data/rgb.md)**: Learn about supported data formats

