#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=160:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eleftherios.ioannou@sheffield.ac.uk
#SBATCH --job-name=SimCLR_Passerines_Kornia_Aves-Back

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

# python scripts/simclr_birdcolour_kornia.py --config config_kornia.yaml

# python scripts/simclr_birdcolour_kornia_color.py --config config_kornia_exp.yaml

python scripts/simclr_birdcolour_kornia02.py --config configs/config_kornia02.yaml
