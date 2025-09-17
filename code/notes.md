# Generating embeddings using ViT/SimCLR

This note summarises ways to extract useful embeddings from a SimCLR-trained ViT backbone, how to use the new `simclr/scripts/generate_embeddings_vit_layers.py`, and trade-offs between different options.

- __Relevant files__
  - `simclr/scripts/simclr_birdcolour_kornia02.py` defines `SimCLRModel` and `SimCLRDataModule`.
  - `simclr/scripts/generate_embeddings.py` generates embeddings from the final backbone output.
  - `simclr/scripts/generate_embeddings_vit_layers.py` (new) extracts intermediate ViT block features with flexible pooling/aggregation.


## 1) Where embeddings come from

- __Backbone__: Torchvision ViT-L/16 (`vit_l_16`) with `hidden_dim = 1024`, classification head removed.
- __Training__: SimCLR contrastive objective applied on the projection head output `z`. The backbone feature `h` (pre-projection) learns representations but the projection head is used only in training.
- __Inference__: For downstream tasks, we typically export features from the backbone (not the projection head). The script `generate_embeddings_vit_layers.py` allows control over which layers/tokens to use.


## 2) Token choices within a ViT block

Given block output tokens `tokens ∈ R^{B×(1+N)×D}` with CLS at index 0 and patch tokens at 1..N:

- __CLS token (`--vit_pool cls`)__
  - Compact, often aligned with classification semantics.

- __Patch mean (`--vit_pool mean`)__
  - Strong for retrieval/semantic similarity and fine-grained patterns; robust.

- __CLS + Patch mean (`--vit_pool cls_mean`)__
  - Combines global and local signals; often stronger.
  - Doubles dimensionality per layer (for ViT-L/16: 2×1024).


## 3) Which layer(s) to use

- __Last layer only (`--vit_last_k 0 --vit_layer -1`)__
  - Most semantic; good default.
- __Earlier single layer (e.g., `--vit_layer 6`)__
  - Captures more local/texture signals. Useful for fine-grained visual details.
- __Aggregate last-k layers (`--vit_last_k K`)__
  - E.g., last 2–4 layers.
  - Choose K with care: larger K → larger or more averaged embeddings; potentially better but heavier.


## 4) Aggregating across layers

Applicable when `--vit_last_k > 0`:

- __Concat (`--vit_aggregate concat`)__
  - Preserves all information across layers; strong baseline.
  - Linear growth in dimensionality with K.

- __Mean (`--vit_aggregate mean`)__
  - Keeps dimensionality fixed.
  - Averages nuances across layers; sometimes slightly weaker than concat.


## 5) Normalization

- __L2 normalization (default on)__: Standard for retrieval/similarity; stabilizes kNN and cosine similarity.
- Optional post-processing (not implemented here but could be helpful):
  - __Feature standardization__: z-score each dimension over the dataset.
  - __PCA/whitening__: Compress and decorrelate features; useful when dimensionality becomes large.


## 6) Dimensionality math (ViT-L/16)

Let `D=1024` per layer:

- Pooling per layer:
  - `cls` or `mean` → D
  - `cls_mean` → 2D
- Aggregation across layers (K layers):
  - `concat` → K×(pooled dim)
  - `mean` → (pooled dim)

Examples:
- `--vit_last_k 4 --vit_pool mean --vit_aggregate concat` → 4×1024 = 4096 dims
- `--vit_last_k 4 --vit_pool cls_mean --vit_aggregate concat` → 4×2048 = 8192 dims
- `--vit_last_k 4 --vit_pool mean --vit_aggregate mean` → 1024 dims


## 7) ResNet fallback

If you load a checkpoint with a non-ViT backbone (e.g., `resnet50`), the script falls back to using the final global feature map (`model.backbone(x).flatten(...)`). Layer hooks are ViT-specific.


## 8) Things to try

- Default (robust retrieval)
  - `--vit_last_k 4 --vit_pool mean --vit_aggregate concat` (L2 on)
  - Produces 4096-D vectors.

- Compact but strong
  - `--vit_last_k 3 --vit_pool mean --vit_aggregate mean` → 1024-D.

- Add global context
  - `--vit_last_k 2 --vit_pool cls_mean --vit_aggregate concat` → 4096-D.

- Earlier-layer texture bias
  - `--vit_last_k 0 --vit_layer 6 --vit_pool mean` → 1024-D.


## 9) How to run

Using the new script `simclr/scripts/generate_embeddings_vit_layers.py`:

```bash
# Robust default (last 4, patch-mean, concat)
python simclr/scripts/generate_embeddings_vit_layers.py \
  --model_path PATH/ckpt.ckpt \
  --config PATH/config.yaml \
  --output_dir OUT_DIR \
  --data_dir DATA_DIR \
  --vit_last_k 4 \
  --vit_pool mean \
  --vit_aggregate concat

# Compact avg across last 3 layers
python simclr/scripts/generate_embeddings_vit_layers.py \
  --model_path PATH/ckpt.ckpt \
  --config PATH/config.yaml \
  --output_dir OUT_DIR \
  --data_dir DATA_DIR \
  --vit_last_k 3 \
  --vit_pool mean \
  --vit_aggregate mean

# Single earlier layer (6), CLS token only
python simclr/scripts/generate_embeddings_vit_layers.py \
  --model_path PATH/ckpt.ckpt \
  --config PATH/config.yaml \
  --output_dir OUT_DIR \
  --data_dir DATA_DIR \
  --vit_last_k 0 \
  --vit_layer 6 \
  --vit_pool cls
```

Outputs:
- CSV at `OUT_DIR/embeddings_vit_layers_*.csv` with columns: `filename, x1..xD`.
- KNN example plot at `OUT_DIR/knn_examples_vit_layers.png`.


## 10) Misc

- Start with: `last_k=4`, `pool=mean`, `aggregate=concat`, L2 on.
- If too large, switch to `aggregate=mean` or smaller `last_k`.
- For fine-grained texture, try a mid-layer (e.g., `vit_layer=6`) with `pool=mean`.
