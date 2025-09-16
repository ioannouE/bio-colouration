# Bio-Colouration AI Analysis

**Unlocking the complexity of organismal colour patterns using AI**

This project provides a comprehensive framework for analyzing and characterizing colour pattern variation within and between bird species using advanced machine learning techniques, particularly contrastive learning with SimCLR.

## 🎯 Project Overview

The repository contains code for analyzing biological colour patterns across three different data modalities:

- **RGB Images**: Standard color images of biological organisms
- **Multispectral Data**: 7-channel data including visible and UV spectrum
- **Hyperspectral Data**: 408-band spectral data from the University of St Andrews Lepidoptera Database

## 🚀 Key Features

### SimCLR Implementation
- Contrastive learning for generating meaningful embeddings
- Support for multiple data types (RGB, multispectral, hyperspectral)
- Advanced data augmentation using the Kornia library
- Cross-modal contrastive learning between RGB and UV channels

### Evaluation & Visualization
- Comprehensive embedding analysis tools
- Statistical metrics (R², Kruskal-Wallis H test)
- Interactive visualizations with image thumbnails
- PCA-transformed embedding analysis

## 🔬 Research Applications

This framework is designed for researchers working on:

- **Biological Pattern Analysis**: Understanding colour variation in organisms
- **Computer Vision**: Developing robust visual representations
- **Comparative Biology**: Analyzing patterns across species
- **Spectral Imaging**: Working with multispectral and hyperspectral data

## 📊 Supported Data Types

| Data Type | Channels | Description |
|-----------|----------|-------------|
| RGB | 3 | Standard color images |
| Multispectral | 7 | RGB (1-3) + UV (4-6) + Segmentation mask (7) |
| Hyperspectral | 408 | Full spectral range 348-809 nm |

## 🛠️ Technology Stack

- **PyTorch** for deep learning
- **Lightly** for self-supervised learning
- **Kornia** for computer vision augmentations
- **Scikit-learn** for analysis and evaluation
- **Rasterio** for geospatial data handling

## 📖 Quick Navigation

- **[Getting Started](getting-started/overview.md)**: Installation and setup guide
- **[Data Types](data/rgb.md)**: Understanding different data modalities
- **[SimCLR Module](simclr/introduction.md)**: Core contrastive learning implementation
- **[Evaluation Tools](eval-vis/overview.md)**: Analysis and visualization utilities
- **[Examples](examples/basic.md)**: Practical usage examples

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{bio-colouration,
  title={Bio-Colouration AI Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/bio-colouration}
}
```

---

Ready to get started? Check out our [installation guide](getting-started/installation.md) or jump into a [quick start tutorial](getting-started/quickstart.md)!
