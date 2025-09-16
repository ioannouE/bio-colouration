# SimCLR Introduction

SimCLR (Simple Contrastive Learning of Visual Representations) is the core machine learning framework used in Phenoscape for generating meaningful embeddings from biological images.

## What is SimCLR?

SimCLR is a self-supervised learning method that learns visual representations by contrasting different augmented views of the same image. It doesn't require labeled data, making it ideal for biological datasets where annotations may be limited or expensive to obtain.

## How SimCLR Works

### 1. Data Augmentation
Two different augmented views are created from each input image using transformations like:
- Random cropping and resizing
- Color jittering (brightness, contrast, saturation, hue)
- Gaussian blur
- Random horizontal/vertical flips

### 2. Encoder Network
A neural network (typically ResNet or Vision Transformer) encodes the augmented images into feature representations.

### 3. Projection Head
A small neural network projects the features into a space where contrastive learning is performed.

### 4. Contrastive Loss
The model learns to:
- **Pull together**: Representations of different views of the same image
- **Push apart**: Representations of different images

## Phenoscape Implementation

Our implementation extends the basic SimCLR framework with several enhancements:

### Multi-Modal Support
- **RGB**: Standard 3-channel images
- **Multispectral**: 7-channel data with UV information
- **Hyperspectral**: 408-band spectral data

### Advanced Augmentations
Using the Kornia library for GPU-accelerated augmentations:
- RandomResizedCrop
- RandomColorJitter
- RandomGrayscale
- RandomGaussianBlur
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation
- RandomPerspective
- RandomThinPlateSpline
- RandomErase
- RandomPosterize
- RandomSharpness

### Cross-Modal Learning
Compare representations across different spectral modalities:
- RGB vs UV channel learning
- Multi-modal fusion strategies
- Modality-specific encoders

## Key Benefits

### For Biological Research
- **No Labels Required**: Learn from unlabeled image collections
- **Robust Features**: Captures biologically relevant patterns
- **Transfer Learning**: Pre-trained models work across species
- **Spectral Analysis**: Leverages UV and hyperspectral information

### Technical Advantages
- **Scalable**: Works with large datasets
- **Flexible**: Supports various backbone architectures
- **Efficient**: GPU-optimized augmentations
- **Extensible**: Easy to add new data modalities

## Architecture Overview

```
Input Image
    ↓
Data Augmentation (2 views)
    ↓
Encoder (ResNet/ViT)
    ↓
Projection Head
    ↓
Contrastive Loss
```

## Training Process

1. **Load Data**: Images organized by species/categories
2. **Augment**: Create two views of each image
3. **Encode**: Pass through backbone network
4. **Project**: Map to contrastive learning space
5. **Contrast**: Compute loss between positive/negative pairs
6. **Optimize**: Update network weights

## Output Embeddings

After training, the encoder produces fixed-size embeddings that:
- Capture semantic similarity between images
- Preserve biological relationships
- Enable downstream analysis and visualization
- Support classification and clustering tasks

## Next Steps

- **[Configuration](configuration.md)**: Learn about training parameters
- **[Training](training.md)**: Detailed training procedures
- **[Data Augmentation](augmentation.md)**: Customizing augmentation strategies
- **[Cross-Modal Learning](cross-modal.md)**: Multi-modal training approaches
