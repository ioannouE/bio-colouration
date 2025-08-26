#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import os
import re
import csv
import yaml
import json
import torch
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Reuse your existing model/data module and plotting util
from simclr_birdcolour_kornia02 import SimCLRModel, SimCLRDataModule, plot_knn_examples


def _extract_vit_blocks(vit_backbone, x, layer_indices):
    """
    Capture outputs (tokens) from specified ViT encoder layers using forward hooks.
    Returns a list of tensors, one per requested layer, shape: (B, 1+N, D).
    """
    feats = {}
    handles = []

    # Torchvision ViT exposes encoder layers as a Sequential
    if not hasattr(vit_backbone, "encoder") or not hasattr(vit_backbone.encoder, "layers"):
        raise RuntimeError("Backbone does not look like a torchvision ViT with encoder.layers")

    layers = vit_backbone.encoder.layers

    def get_hook(idx):
        def hook(_m, _inp, out):
            feats[idx] = out
        return hook

    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Requested ViT layer index {idx} out of range [0, {len(layers)-1}]")
        handles.append(layers[idx].register_forward_hook(get_hook(idx)))

    # Run forward once; hooks will populate feats
    _ = vit_backbone(x)

    for h in handles:
        h.remove()

    # preserve requested order
    return [feats[i] for i in layer_indices]


def _pool_tokens(tokens, pool="cls"):
    """
    tokens: (B, 1+N, D) with CLS at index 0.
    pool in {'cls','mean','cls_mean'}.
    """
    if pool == "cls":
        return tokens[:, 0]  # (B, D)
    elif pool == "mean":
        return tokens[:, 1:].mean(dim=1)  # (B, D)
    elif pool == "cls_mean":
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)  # (B, 2D)
    else:
        raise ValueError(f"Unsupported pool={pool}")


def _print_vit_structure(backbone, verbose: bool = False):
    """Print a summary of the ViT backbone, including encoder blocks.

    If verbose=True, also print the full backbone repr.
    """
    print("Backbone type:", type(backbone).__name__)
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layers"):
        layers = backbone.encoder.layers
        print(f"ViT encoder has {len(layers)} blocks:")
        for i, layer in enumerate(layers):
            # Print class name and a short repr
            print(f"  [#{i:02d}] {layer.__class__.__name__}: {layer}")
    else:
        print("Backbone does not expose encoder.layers (not a torchvision ViT or unexpected API)")

    if verbose:
        print("\nFull backbone definition:\n", backbone)


def generate_embeddings_vit_layers(
    model,
    dataloader,
    vit_layer: int = -1,            # negative index from end; used when vit_last_k == 0
    vit_last_k: int = 4,            # if >0, use last-k blocks
    vit_pool: str = "mean",         # 'cls' | 'mean' | 'cls_mean'
    vit_aggregate: str = "concat",  # 'concat' | 'mean' across layers when multiple
    l2_normalize: bool = True,
):
    """
    Generate embeddings with ViT intermediate layers.
    Falls back to final backbone output if the backbone is not ViT.
    Returns (embeddings: np.ndarray [N, D], filenames: List[str]).
    """
    from tqdm import tqdm
    embeddings = []
    filenames = []

    is_vit = hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layers")

    with torch.no_grad():
        for img, _, fnames in tqdm(dataloader, desc="Generating embeddings (ViT layers)", unit="batch"):
            img = img.to(model.device)

            if is_vit:
                num_layers = len(model.backbone.encoder.layers)
                if vit_last_k and vit_last_k > 0:
                    layer_indices = list(range(num_layers - vit_last_k, num_layers))
                else:
                    li = vit_layer if vit_layer >= 0 else num_layers + vit_layer  # handle negatives
                    layer_indices = [li]

                tokens_list = _extract_vit_blocks(model.backbone, img, layer_indices)
                pooled_list = [_pool_tokens(t, pool=vit_pool) for t in tokens_list]

                if len(pooled_list) > 1:
                    if vit_aggregate == "concat":
                        emb = torch.cat(pooled_list, dim=-1)
                    elif vit_aggregate == "mean":
                        emb = torch.stack(pooled_list, dim=0).mean(dim=0)
                    else:
                        raise ValueError("vit_aggregate must be one of {'concat','mean'}")
                else:
                    emb = pooled_list[0]
            else:
                # Non-ViT backbone (e.g., ResNet) -> use existing behavior
                emb = model.backbone(img).flatten(start_dim=1)

            if l2_normalize:
                emb = torch.nn.functional.normalize(emb, dim=1)

            embeddings.append(emb.cpu())
            filenames.extend(fnames)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings, filenames


def generate_embeddings_from_model(args):
    """
    Mirror your existing flow but route through the new generator above.
    """
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading data from: {args.data_dir}")

    # Override data_dir and avoid multiprocessing for predict
    config["data"]["data_dir"] = args.data_dir
    if "dataloader" not in config:
        config["dataloader"] = {}
    config["dataloader"]["num_workers"] = 0
    print("Set num_workers=0 to avoid shared memory issues during embedding generation")

    # Data
    data_module = SimCLRDataModule(config)
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    # Model
    model_config = {
        "scheduler": config["model"]["scheduler"],
        "optimizer": config["model"]["optimizer"],
        "criterion": config["model"]["criterion"],
        "backbone": config["model"]["backbone"],
        "weights": config["model"]["weights"],
    }

    print(f"Loading model from checkpoint: {args.model_path}")
    model = SimCLRModel.load_from_checkpoint(args.model_path, config=model_config)
    model.eval()

    # Optional: inspect ViT and exit
    if getattr(args, "inspect_vit", False):
        _print_vit_structure(model.backbone, verbose=getattr(args, "inspect_verbose", False))
        return

    # Generate embeddings with ViT intermediate layers
    print("Generating embeddings with ViT layer controls...")
    embeddings, filenames = generate_embeddings_vit_layers(
        model,
        dataloader,
        vit_layer=args.vit_layer,
        vit_last_k=args.vit_last_k,
        vit_pool=args.vit_pool,
        vit_aggregate=args.vit_aggregate,
        l2_normalize=not args.no_l2_normalize,
    )
    print("Embeddings generated.")

    # Save embeddings
    os.makedirs(args.output_dir, exist_ok=True)
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f"x{i+1}" for i in range(len(df_embeddings.columns))]
    df_filenames = pd.DataFrame(filenames, columns=["filename"])
    df = pd.concat([df_filenames, df_embeddings], axis=1)

    # Timestamp with config filename
    config_filename = os.path.splitext(os.path.basename(args.config))[0]
    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec="minutes")
    version = f"{config_filename}_{''.join(re.split(r'[-:]', date_str))}"
    output_filename = f"embeddings_vit_layers_{version}.csv"

    out_csv = os.path.join(args.output_dir, output_filename)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8", float_format="%.15f")
    print(f"Saved embeddings to: {out_csv}")

    # KNN plot
    try:
        print("Creating sample KNN plots...")
        plot_knn_examples(embeddings, filenames, args.data_dir)
        plt.savefig(os.path.join(args.output_dir, "knn_examples_vit_layers.png"))
        print("Saved KNN plot.")
    except Exception as e:
        print(f"Skipping KNN plot due to error: {e}")


def get_args():
    parser = ArgumentParser(description="Generate embeddings using intermediate ViT layers")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file used for training")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the embeddings")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")

    # ViT controls
    parser.add_argument("--vit_layer", type=int, default=-1,
                        help="Specific ViT encoder block index (negative for from-end, e.g. -1 = last); used when --vit_last_k=0")
    parser.add_argument("--vit_last_k", type=int, default=4,
                        help="If >0, aggregate the last-k ViT blocks (overrides --vit_layer)")
    parser.add_argument("--vit_pool", type=str, default="mean", choices=["cls", "mean", "cls_mean"],
                        help="Pooling strategy for tokens within a block")
    parser.add_argument("--vit_aggregate", type=str, default="concat", choices=["concat", "mean"],
                        help="How to combine multiple layers when --vit_last_k>0")
    parser.add_argument("--no_l2_normalize", action="store_true", help="Disable L2 normalization of embeddings")

    # Inspection controls
    parser.add_argument("--inspect_vit", action="store_true", help="Print the ViT architecture (encoder blocks) and exit")
    parser.add_argument("--inspect_verbose", action="store_true", help="Also print the full backbone repr when inspecting")

    return parser.parse_args()


def main():
    args = get_args()
    generate_embeddings_from_model(args)
    print("\nEmbeddings (ViT layers) generation complete!")


if __name__ == "__main__":
    main()