#!/bin/bash

# Make sure the script exits on error
set -e

# Set your paths and parameters
CSV_PATH="./data/your_embeddings.csv"
IMAGE_DIR="./data/images"
OUTPUT_DIR="./results"
LABELS="color shape"
EMB_NUM=1024
EMB_PREFIX="x"
PCA_NC=50
PLOT_NC=6
SUBSET_SIZE=200

# Run the evaluation script
python evaluate.py \
  --csv_path "$CSV_PATH" \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --labels $LABELS \
  --emb_num "$EMB_NUM" \
  --emb_prefix "$EMB_PREFIX" \
  --pca_nc "$PCA_NC" \
  --plot_nc "$PLOT_NC" \
  --subset_size "$SUBSET_SIZE"
