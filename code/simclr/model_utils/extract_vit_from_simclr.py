#!/usr/bin/env python3
"""
Extract ViT backbone weights from a SimCLR Lightning checkpoint.
- Loads your trained SimCLR model (ViT-L/16 backbone) via Lightning.
- Extracts only the ViT backbone state_dict (excluding projection head).
- Optionally prints key names and validates basic SAM compatibility.
- Saves the backbone weights to a .pth file for later reuse.

Usage:
  python simclr/scripts/extract_vit_from_simclr.py \
      --ckpt /path/to/checkpoint.ckpt \
      --config /path/to/config.yaml \
      --out /path/to/vit_backbone.pth \
      [--print-keys] [--verify-sam]

Note:
- This saves raw torchvision ViT-L/16 keys (e.g., 'conv_proj', 'encoder.layers.*').
- For SAM integration, a later step will remap these keys and interpolate positional embeddings.
"""

import argparse
import json
import os
from typing import Dict, Any

import torch
import yaml

# Import your Lightning SimCLR model (same style as other scripts)
from simclr_birdcolour_kornia02 import SimCLRModel


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Build the minimal model_config expected by SimCLRModel.load_from_checkpoint
    model_cfg = {
        "scheduler": cfg["model"]["scheduler"],
        "optimizer": cfg["model"]["optimizer"],
        "criterion": cfg["model"]["criterion"],
        "backbone": cfg["model"]["backbone"],
        "weights": cfg["model"].get("weights"),
    }
    return model_cfg


def _gather_meta(backbone: torch.nn.Module) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    # Embed dim (from final norm or first ln layer)
    try:
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layers"):
            meta["num_layers"] = len(backbone.encoder.layers)
            meta["embed_dim"] = int(backbone.encoder.layers[0].ln_1.weight.shape[0])
            meta["final_norm_dim"] = int(backbone.encoder.ln.weight.shape[0])
        else:
            meta["num_layers"] = None
    except Exception:
        pass

    # Patch/chan from conv_proj
    try:
        w = backbone.conv_proj.weight  # (embed_dim, in_chans, patch, patch)
        meta["in_chans"] = int(w.shape[1])
        meta["patch_size"] = int(w.shape[-1])
    except Exception:
        pass

    # Positional embedding seq length
    try:
        pe = backbone.encoder.pos_embedding  # (1, 1+N, D)
        meta["pos_embed_shape"] = list(pe.shape)
        seq_len = int(pe.shape[1])
        meta["seq_len"] = seq_len
        meta["has_cls_token"] = True  # torchvision ViT includes CLS
        meta["num_patches"] = seq_len - 1
    except Exception:
        pass

    return meta


def _verify_for_sam(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small report on basic SAM compatibility expectations."""
    report = {}
    report["expect_vit_L_embed_dim_1024"] = (meta.get("embed_dim") == 1024)
    report["expect_patch_size_16"] = (meta.get("patch_size") == 16)
    report["expect_in_chans_3"] = (meta.get("in_chans") == 3)
    # Training image size often 224 -> seq_len = 1 + 14*14
    if "seq_len" in meta and "patch_size" in meta:
        report["train_grid_size"] = int((meta["seq_len"] - 1) ** 0.5) if meta["seq_len"] else None
    return report


def extract_and_save(ckpt: str, config_path: str, out_path: str, print_keys: bool, verify_sam: bool):
    # Load model
    model_config = _load_config(config_path)
    print("Loading SimCLR Lightning checkpoint:", ckpt)
    model = SimCLRModel.load_from_checkpoint(ckpt, config=model_config, map_location="cpu")
    model.eval()

    # Extract ViT backbone weights only
    vit_state = model.backbone.state_dict()

    # Optional introspection
    meta = _gather_meta(model.backbone)
    print("Backbone meta:")
    print(json.dumps(meta, indent=2))

    if verify_sam:
        report = _verify_for_sam(meta)
        print("\nSAM compatibility (basic) checks:")
        print(json.dumps(report, indent=2))

    if print_keys:
        print("\nState dict keys (ViT backbone):")
        for k in vit_state.keys():
            print(" ", k)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Save as raw torchvision ViT backbone
    torch.save(vit_state, out_path)
    print(f"\nSaved ViT backbone weights to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Extract ViT-L/16 backbone from SimCLR Lightning checkpoint")
    p.add_argument("--ckpt", required=True, type=str, help="Path to Lightning .ckpt file")
    p.add_argument("--config", required=True, type=str, help="Path to YAML config used for training")
    p.add_argument("--out", required=True, type=str, help="Output path for extracted ViT backbone .pth")
    p.add_argument("--print-keys", action="store_true", help="Print backbone state_dict keys")
    p.add_argument("--verify-sam", action="store_true", help="Run basic checks for SAM compatibility")
    args = p.parse_args()

    extract_and_save(
        ckpt=args.ckpt,
        config_path=args.config,
        out_path=args.out,
        print_keys=args.print_keys,
        verify_sam=args.verify_sam,
    )


if __name__ == "__main__":
    main()
