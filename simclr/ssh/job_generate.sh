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
#SBATCH --job-name=SimCLR_Passerines_generateFrom3809137_Back

module load Anaconda3/2019.07
source activate simclr
module load cuDNN/8.0.4.30-CUDA-11.1.1 

# python scripts/generate_embeddings.py \
#         --model_path /shared/cooney_lab/Shared/Eleftherios-Ioannou/SimCLR/different_views/simclr/TIMEOUT_config_kornia02_20250408T1408/checkpoints/epoch\=104-step\=254100.ckpt \
#         --config configs/config_kornia02.yaml \
#         --output_dir /shared/cooney_lab/Shared/Eleftherios-Ioannou/SimCLR/different_views/simclr/TIMEOUT_config_kornia02_20250408T1408/

python scripts/generate_embeddings.py \
        --model_path /shared/cooney_lab/Shared/Eleftherios-Ioannou/SimCLR/different_views/simclr/config_kornia02_20250414T0933/checkpoints/epoch\=199-step\=322600.ckpt \
        --config configs/config_kornia02.yaml \
        --output_dir /shared/cooney_lab/Shared/Eleftherios-Ioannou/SimCLR/different_views/simclr/config_kornia02_20250414T0933/
