# Installation

This guide will help you set up Phenoscape on your system.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

## Environment Setup

### Option 1: Conda Environment (Recommended)

Create a new conda environment with the required packages:

```bash
conda create -n phenoscape python=3.8
conda activate phenoscape
```

### Option 2: Virtual Environment

```bash
python -m venv phenoscape-env
source phenoscape-env/bin/activate  # On Windows: phenoscape-env\Scripts\activate
```

## Install Dependencies

The project requires the following key packages:

```bash
pip install torch==2.6.0 torchvision==0.21.0
pip install albumentations==1.4.20
pip install matplotlib==3.10.1
pip install opencv-python==4.11.0.86
pip install scikit-image==0.25.2
pip install scikit-learn==1.6.1
pip install tensorboard==2.19.0
pip install timm==0.4.5
pip install tqdm==4.67.1
pip install pillow==11.1.0
pip install PyYAML==6.0.2
pip install rasterio==1.4.3
```

Or install from the requirements file:

```bash
cd simclr
pip install -r requirements.txt
```

### Additional Dependencies for Specialized Data

For multispectral and hyperspectral data processing:

```bash
pip install mmcv-full==1.5.0
pip install mmsegmentation==0.27.0
```

For SAM-2 segmentation:

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## HPC Setup (Bessemer Cluster)

If you're using the Bessemer HPC cluster:

```bash
module load Anaconda3/2019.07
module load cuDNN/8.0.4.30-CUDA-11.1.1
```

## Verify Installation

Test your installation by running:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Clone the Repository

```bash
git clone https://github.com/your-username/bio-colouration.git
cd bio-colouration
```

## Next Steps

- **[Quick Start](quickstart.md)**: Run your first analysis
- **[Configuration](../simclr/configuration.md)**: Learn about configuration options
- **[Data Types](../data/rgb.md)**: Understand supported data formats

## Troubleshooting

### Common Issues

**CUDA Out of Memory**: Reduce batch size in configuration files
**Import Errors**: Ensure all dependencies are installed in the correct environment
**Permission Errors**: Check file permissions and directory access

For more help, see the [Performance Optimization](../advanced/optimization.md) guide.
