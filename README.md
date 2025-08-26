# Unlocking the complexity of organismal colour patterns using AI

## Code overview

The repository contains code for analysing and characterising the colour pattern variation within and between bird species.
The code is split into several submodules:

- `simclr`: SimCLR implementation generating embeddings using the SimCLR method that uses contrastive learning
- `dino`: Code for generating embeddings using the DINOv2 method that uses self-supervised learning
- `mae`: Code for generating embeddings using the MAE that is based on reconstruction


## Datasets

The datasets used in this repository are:

### RGB
- `Passerines-Segmented-Back-RGB-PNG`
- `Passerines-Segmented-Belly-RGB-PNG`
- `Passerines-Segmented-Back-RGB-PNG`
- `Passerines-Warped-Back-RGB-PNG`

### multispectral
- `Multispec_test_images_5000`

### hyperspectral
- `data/hyperspectral/lepidoptera`

## Setup

To run the code on HPC (Bessemer cluster):

```bash
module load Anaconda3/2019.07
module load cuDNN/8.0.4.30-CUDA-11.1.1 
```


The conda environment created has the following packages:

- `albumentations`              1.4.20
- `matplotlib`                  3.10.1
- `mmcv-full`                   1.5.0
- `mmsegmentation`              0.27.0
- `opencv-python`               4.11.0.86
- `opencv-python-headless`      4.11.0.86
- `SAM-2`                       1.0
- `scikit-learn`                1.6.1
- `tensorboard`                 2.19.0
- `timm`                        0.4.5
- `torch`                       2.6.0
- `torchvision`                 0.21.0
- `tqdm`                        4.67.1
