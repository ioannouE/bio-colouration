# SimCLR

The code implements the SimCLR method for generating embeddings using contrastive learning. It is based on the SimCLR implementation from the lightly library and on the code from the [RSE-bird-colour](https://github.com/christophercooney/RSE-bird-colour) repository.


## Contents

### Scripts
- Implementation of the SimCLR method
- Scripts for hyperparameter optimisation using RayTune, and grid search
- Scripts for generating embeddings from a trained model
- Scripts using ViT as a backbone and the kornia library for data augmentation

### configs
- Configuration files for the SimCLR method

### eval
- Scripts for evaluating the embeddings

### plots
- Generate umap plot of the embeddings
- Use PageRank algorithm to create a graph of the embeddings and identify the most influential and least influential images
- Use different CV metrics (LPIPS, SSIM, PSNR) to generate dissimilarity matrix between images and visualise using UMAP and t-SNE 

### visualisations 
- Plot the embeddings using images as thumbnails


## Setup

An initial implementation of the SimCLR method is (created in the `RSE-bird-colour` repository) in
`scripts/simclr_birdcolour.py`.  This is largely based on the [Lightly
SSL SimCLR
tutorial](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html).

### Set up Python environment

Set up a Python environment using conda or by creating a virtual environment using conda or venv.

Activate the environment and load required packages:

```
pip install -r requirements.txt
```

### Configuration

The workflow can be configured by modifying parameters in
`config.yaml`.  Current configurable parameters:

- `data_dir` - image data location
- `out_dir` - output directory (also used for logging)
- `backbone` - pretrained model backbone to use
- `weights` - whether to use pretrained weights (`null` or `'DEFAULT'`)
- `lr` - learning rate for optimizer
- `T_max` - maximum iterations for learning rate scheduler
- `temperature` - parameter for constrastive loss
- `seed` - value for setting seeds for pseudo-random generators
- `num_workers` - number of threads to use when loading data
- `batch_size` - number of images in a batch, adjust value according to available GPU memory
- `deterministic` - to ensure full reproducibility from run to run
- `accelerator` - `'cpu'`, `'gpu'` or `'auto'`
- `devices` - number of devices to train on
- `max_epochs` - when to stop training
- `log_every_n_steps` - how often to add logging

#### Augmentations

The configuration file also contains parameters for the transforms applied
during data augmentation. The parameters will be passed into Lightly's
[`SimCLRTransform`](https://docs.lightly.ai/self-supervised-learning/lightly.transforms.html#lightly.transforms.simclr_transform.SimCLRTransform).


Added augmenations using the `kornia` library:

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


### Running the SimCLR implementation

The implementation can be run from the command line:

```
python scripts/simclr_birdcolour.py
```

The implementation assumes that `config.yaml` is in the root directory
as in this repository. To use a configuration in a different location,
specify the path using the `-c`/`--config` flag:

```
python scripts/simclr_birdcolour.py -c path/to/my/config.yaml
```

Usage:
```
python scripts/simclr_birdcolour.py --help
usage: simclr_birdcolour.py [-h] [-c CONFIG]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to configuration file
```


### Multispectral data

The multispectral data is processed using the `scripts/multispectral_data.py` script.

To train SimCLR on multispectral data:

```bash
python scripts/simclr_kornia_spectral.py
```

Flags:
- `--config`: Path to configuration file
- `--rgb-only`: Use only RGB channels (first 3 layers) and ignore UV data
- `--usml`: Use tetrahedral colour space (USML)
- `--model-checkpoint`: Path to pretrained MultispectralClassifier checkpoint to use as backbone
- `--visualize`: Visualize augmentations applied to random samples from the dataset
- `--loss-type`: Loss function to use (global, local, or combined)
  - `global`: Use global loss only, as normal
  - `local`: Use local loss only - applies the contrastive loss on the local patches
  - `combined`: Use both global and local loss
- `--global-weight`: Weight for global loss 
- `--local-weight`: Weight for local loss


### Exploring cross-modal contrastive loss

SimCLR with multispectral data can be used to explore cross-modal contrastive learning. Instead of contrasting an image with its augmented view, we can contrast the RGB channels with the UV channels. 
This can be done using the `scripts/simclr_birdcolour_kornia_spectral_multimodal.py` script.


#### Different cross modal fusion strategies

The cross-modal contrastive loss can be combined with different fusion strategies:

- `concat`: Concatenate the RGB and UV channels
- `separate`: Use separate encoders for RGB and UV channels
- `attention`: Use attention to fuse the RGB and UV channels

To use a different fusion strategy, set the `--fusion-type` flag:

```bash
python scripts/simclr_birdcolour_kornia_spectral_multimodal.py --fusion-type concat
```

#### Modality-specific encoders

The cross-modal contrastive loss can be combined with modality-specific encoders. This means that we can use different encoders for the RGB and UV channels. 

To use modality-specific encoders, set the `--use-modality-specific` flag:

```bash
python scripts/simclr_birdcolour_kornia_spectral_multimodal.py --use-modality-specific
```
  

### Hyperspectral data

Dataset is located at `data/hyperspectral/lepidoptera` and processed using the `scripts/hyperspectral_data.py` script.
Dataset is retrieved from the approach [A computational neuroscience framework for quantifying warning signals.](https://arts.st-andrews.ac.uk/lepidoptera/index.html) 

The hyperspectral data requires files in the format `.bil` and `.bil.hdr`. The data has 408 bands.

To train SimCLR on hyperspectral data:

```bash
python scripts/simclr_birdcolour_kornia_hyperspectral.py
```

Flags:
- `--config`: Path to configuration file
- `--rgb-only`: Use only RGB channels (first 3 layers) and ignore UV data
- `--sample-bands`: Comma-separated list of band indices to sample from hyperspectral data (e.g., '0,50,100,150,200')
- `--model-checkpoint`: Path to pretrained MultispectralClassifier checkpoint to use as backbone
- `--visualize`: Visualize augmentations applied to random samples from the dataset
- `--loss-type`: Loss function to use (global, local, or combined)
- `--global-weight`: Weight for global NTXentLoss
- `--local-weight`: Weight for local NTXentLoss
- `--test-embeddings-only`: Skip training and only generate embeddings for testing


#### Reading hyperspectral data

The hyperspectral data is read using the `scripts/hyperspectral_data.py` script.
This script uses the `read_iml_hyp.py` script that is a re-implementation of the `read_iml_hyp.m` script from the original paper.

##### RGB 
For transforming to RGB, wavelengths closest to 450nm, 550nm, and 650nm are used.

##### Sample Bands 
To sample specific bands, use the `--sample-bands` flag. For example, to sample bands 0, 50, 100, 150, and 200, use the following command:

```bash
python scripts/hyperspectral_data.py --sample-bands 0,50,100,150,200
```

Otherwise, all the bands are used.
