# Unlocking the complexity of organismal colour patterns using AI

## Code overview

The repository contains code for analysing and characterising the colour pattern variation within and between bird species.
The code is split into several submodules:

- `simclr`: SimCLR implementation generating embeddings using the SimCLR method that uses contrastive learning
- `eval_vis`: Evaluation utilities and visualisation tools for visualising the embeddings


## Data
The code supports three types of data: RGB, multispectral and hyperspectral.

### RGB
- Standard images of any biological organism are supported. For best results, the images should be segmented into the area of interest (e.g. the body of the bird, removing any background).

### multispectral
- Multispectral images of any biological organism are supported. The multispectral image data should be in the form of `.tif` files. The format currently supported is 7-channel data where the channels are: 1-3 channels: Visible (Red, Green, Blue), 4-6 channels: Ultraviolet (UV), 7th channel: Binary Segmentation mask.

### hyperspectral
- Hyperspectral images of any biological organism are supported. The data currently supported is in the format of the (University of St Andrews Lepidoptera Hyperspectral Database)[https://arts.st-andrews.ac.uk/lepidoptera/]. These are `.bil` files along with `.hdr` files. The scans of Lepidoptera in this database contain 408 spectral bands equally distributed between 348 nm and 809 nm (the common exact wavelength resolution is provided in the ‘.bil.hdr’ file associated with each ‘.bil’ scan).


## Setup

The conda environment created has the following packages:

- `albumentations`              1.4.20
- `matplotlib`                  3.10.1
- `mmcv-full`                   1.5.0
- `mmsegmentation`              0.27.0
- `opencv-python`               4.11.0.86
- `opencv-python-headless`      4.11.0.86
- `SAM-2`                       1.0
- `scikit-image`                0.25.2
- `scikit-learn`                1.6.1
- `tensorboard`                 2.19.0
- `timm`                        0.4.5
- `torch`                       2.6.0
- `torchvision`                 0.21.0
- `tqdm`                        4.67.1
- `pillow`                      11.1.0
- `protobuf`                    5.29.3
- `PyYAML`                      6.0.2
- `rasterio`                    1.4.3


To run the code on HPC (Bessemer cluster):

```bash
module load Anaconda3/2019.07
module load cuDNN/8.0.4.30-CUDA-11.1.1 
```

