# Phenoscape: A Contrastive Representation Learning Framework Across the Spectral Domain

**Phenoscape** is a comprehensive framework for analyzing organismal colour patterns using artificial intelligence. This project focuses on understanding colour pattern variation within and between biological species through modern machine learning techniques.


## What is Phenoscape?

Phenoscape combines state-of-the-art computer vision and contrastive learning methods to analyze biological colour patterns across multiple spectral modalities. The framework is built around SimCLR (Simple Contrastive Learning of Visual Representations) and provides tools for both training models and evaluating embeddings.



## Key Features

### Multi-Modal Data Support
- **RGB Images**: Standard colour images of biological organisms
- **Multispectral Data**: 7-channel data including visible and ultraviolet spectrum
- **Hyperspectral Data**: 408-band spectral imaging from specialised databases


### SimCLR Implementation
- Contrastive learning for generating meaningful embeddings
- Support for multiple data types (RGB, multispectral, hyperspectral)
- Advanced data augmentation using the Kornia library
- Cross-modal contrastive learning between RGB and UV channels

### Evaluation & Visualization
- Comprehensive embedding analysis tools
- Statistical metrics (Silhouette score, Variance explained)
- Interactive visualisations with image thumbnails (UMAP, t-SNE)
- PCA-transformed embedding analysis

## Research Applications

This framework is designed for researchers working on:

- **Biological Pattern Analysis**: Understanding colour variation in organisms
- **Computer Vision**: Developing robust visual representations for biological data
- **Comparative Biology**: Analyzing phenotypic variation across species
- **Spectral Imaging**: Working with multispectral and hyperspectral data

## Supported Data Types

| Data Type | Channels | Description |
|-----------|----------|-------------|
| RGB | 3 | Standard color images |
| Multispectral | 7 | RGB (1-3) + UV (4-6) + Segmentation mask (7) |
| Hyperspectral | 408 | Full spectral range 348-809 nm |

## Technology Stack

- **PyTorch** for deep learning
- **Lightly** for self-supervised learning
- **Kornia** for computer vision augmentations
- **Scikit-learn** for analysis and evaluation
- **Rasterio** for geospatial data handling

## Quick Navigation

- **[Getting Started](getting-started/quickstart.md)**: Installation and setup guide
- **[Data Types](data/rgb.md)**: Understanding different data modalities
- **[SimCLR Module](simclr/introduction.md)**: Core contrastive learning implementation
- **[Evaluation Tools](eval/overview.md)**: Analysis and visualization utilities
- **[Examples](examples/example_rgb.md)**: Practical usage examples


## Acknowledgements

This work was supported by the BBSRC: [Unlocking the complexity of organismal colour patterns using AI](https://gtr.ukri.org/projects?ref=BB%2FY513830%2F1). The toolkit is built on top of Google's SimCLR implementation.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{phenoscape2025,
  title={Phenoscape: A Contrastive Representation Learning Framework Across the Spectral Domain},
  author={Cooney Lab},
  year={2025},
  url={https://github.com/ioannouE/bio-colouration}
}
```

---

Ready to get started? Check out our [installation guide](getting-started/installation.md) or jump into a [quick start tutorial](getting-started/quickstart.md)!
