#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_Multispectral5k_allChannels_FitResize

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

# python scripts/simclr_birdcolour_kornia_spectral.py -c config_kornia_spectral.yaml --model-checkpoint lightning_logs/version_3777301/checkpoints/best-checkpoint.ckpt 

python scripts/simclr_birdcolour_kornia_spectral.py -c configs/config_kornia_spectral.yaml

# python scripts/simclr_birdcolour_kornia_vit.py -c configs/config_kornia_spectral.yaml --rgb-only

# python scripts/simclr_birdcolour_kornia_spectral_multimodal.py \
#         -c configs/config_kornia_spectral.yaml \
#         --loss-type crossmodal \
#         --cross-modal-weight 2.0 \
#         --use-modality-specific \
#         --fusion-type attention

# python scripts/simclr_birdcolour_kornia_spectral.py \
#                 -c configs/config_kornia_spectral.yaml \
#                 --channels 0 1 5 2