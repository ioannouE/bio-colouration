#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_HyperspectralData_sampleBands_0,100,200-CropLast

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 


python scripts/simclr_birdcolour_kornia_hyperspectral.py \
            -c configs/config_kornia_hyperspectral.yaml \
            --sample-bands 0,100,200 \
            --augmentation-strategy 'crop'


# python scripts/simclr_birdcolour_kornia_hyperspectral.py \
#             -c configs/config_kornia_hyperspectral.yaml \
#             --rgb-only 