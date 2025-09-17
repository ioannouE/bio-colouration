import os
import sys
import argparse
import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ensure local imports from this scripts directory work regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Local SimCLR model (keep consistent with generate_embeddings.py)
from simclr_birdcolour_kornia02 import SimCLRModel  # we use the backbone for features

# Optional: ImageNet normalization constants
try:
    from lightly.transforms import utils as lightly_utils
    IMAGENET_MEAN = lightly_utils.IMAGENET_NORMALIZE["mean"]
    IMAGENET_STD = lightly_utils.IMAGENET_NORMALIZE["std"]
except Exception:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

import torchvision.transforms as T
import yaml

# LPIPS (pip install lpips)
try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


class BAPPS2AFCDataset(Dataset):
    """BAPPS 2AFC triplet dataset (ref, p0, p1, judge).

    Expects directory structure like:
      subset_dir/
        ref/
        p0/
        p1/
        judge/
    Files are sorted and matched by index (as in LPIPS repo).
    """

    def __init__(self, subset_dir: str, transform_lpips: Optional[T.Compose] = None,
                 transform_simclr: Optional[T.Compose] = None):
        self.subset_dir = subset_dir
        self.dir_ref = os.path.join(subset_dir, "ref")
        self.dir_p0 = os.path.join(subset_dir, "p0")
        self.dir_p1 = os.path.join(subset_dir, "p1")
        self.dir_judge = os.path.join(subset_dir, "judge")

        def sorted_files(d: str, exts: Tuple[str, ...]) -> List[str]:
            paths = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(d, f"*.{ext}")))
            return sorted(paths)

        self.ref_paths = sorted_files(self.dir_ref, ("png", "jpg", "jpeg", "bmp"))
        self.p0_paths = sorted_files(self.dir_p0, ("png", "jpg", "jpeg", "bmp"))
        self.p1_paths = sorted_files(self.dir_p1, ("png", "jpg", "jpeg", "bmp"))
        self.judge_paths = sorted_files(self.dir_judge, ("npy",))

        n = len(self.ref_paths)
        assert len(self.p0_paths) == n and len(self.p1_paths) == n and len(self.judge_paths) == n, \
            f"Mismatched counts in {subset_dir}"

        self.transform_lpips = transform_lpips
        self.transform_simclr = transform_simclr

    def __len__(self) -> int:
        return len(self.ref_paths)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        ref = self._load_image(self.ref_paths[idx])
        p0 = self._load_image(self.p0_paths[idx])
        p1 = self._load_image(self.p1_paths[idx])
        judge = float(np.load(self.judge_paths[idx]).astype(np.float32).reshape(-1)[0])  # fraction in [0,1]

        sample = {
            "judge": judge,
        }

        if self.transform_lpips is not None:
            sample["ref_lpips"] = self.transform_lpips(ref)
            sample["p0_lpips"] = self.transform_lpips(p0)
            sample["p1_lpips"] = self.transform_lpips(p1)

        if self.transform_simclr is not None:
            sample["ref_simclr"] = self.transform_simclr(ref)
            sample["p0_simclr"] = self.transform_simclr(p0)
            sample["p1_simclr"] = self.transform_simclr(p1)

        return sample


def get_device(prefer: Optional[str] = None) -> torch.device:
    # prefer can be None, 'cpu', 'cuda', 'mps'
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_simclr_model(ckpt_path: str, config_path: str, device: torch.device) -> SimCLRModel:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = {
        "scheduler": cfg["model"]["scheduler"],
        "optimizer": cfg["model"]["optimizer"],
        "criterion": cfg["model"]["criterion"],
        "backbone": cfg["model"]["backbone"],
        "weights": cfg["model"]["weights"],
    }
    model = SimCLRModel.load_from_checkpoint(ckpt_path, config=model_cfg)
    model.eval()
    model = model.to(device)
    return model


def simclr_feature_extractor(model: SimCLRModel, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    # Returns L2-normalized backbone features
    with torch.no_grad():
        feats = model.backbone(x).flatten(1)
        if normalize:
            feats = F.normalize(feats, dim=1)
    return feats


def evaluate_simclr_2afc(subset_dir: str, model: SimCLRModel, device: torch.device,
                          input_size: int, batch_size: int, num_workers: int,
                          distance: str = "cosine", normalize: bool = True) -> float:
    t_simclr = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds = BAPPS2AFCDataset(subset_dir, transform_simclr=t_simclr)

    def collate_fn(batch):
        refs = torch.stack([b["ref_simclr"] for b in batch])
        p0s = torch.stack([b["p0_simclr"] for b in batch])
        p1s = torch.stack([b["p1_simclr"] for b in batch])
        judges = torch.tensor([b["judge"] for b in batch], dtype=torch.float32)
        return refs, p0s, p1s, judges

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    total = 0.0
    count = 0
    for refs, p0s, p1s, judges in dl:
        refs = refs.to(device)
        p0s = p0s.to(device)
        p1s = p1s.to(device)

        z_ref = simclr_feature_extractor(model, refs, normalize=normalize)
        z_p0 = simclr_feature_extractor(model, p0s, normalize=normalize)
        z_p1 = simclr_feature_extractor(model, p1s, normalize=normalize)

        # compute distances
        if distance == "cosine":
            d0 = 1.0 - F.cosine_similarity(z_ref, z_p0, dim=1)
            d1 = 1.0 - F.cosine_similarity(z_ref, z_p1, dim=1)
        elif distance == "euclidean":
            d0 = F.pairwise_distance(z_ref, z_p0, p=2)
            d1 = F.pairwise_distance(z_ref, z_p1, p=2)
        else:
            raise ValueError(f"Unknown distance: {distance}")

        choose_p1 = (d1 < d0).float().cpu()
        judges = judges.cpu()

        # fractional 2AFC score: if choose p1, score=judge; else score=1-judge
        scores = choose_p1 * judges + (1.0 - choose_p1) * (1.0 - judges)
        total += scores.sum().item()
        count += scores.numel()

    return float(total / max(count, 1))


def evaluate_lpips_2afc(subset_dir: str, device: torch.device, batch_size: int, num_workers: int,
                         lpips_net: str = "alex", lpips_size: int = 256) -> float:
    if lpips_lib is None:
        raise RuntimeError("lpips is not installed. Please `pip install lpips`. ")

    # LPIPS expects inputs in [0,1] if normalize=True is passed to forward
    t_ops = []
    if lpips_size and lpips_size > 0:
        t_ops.append(T.Resize((lpips_size, lpips_size), interpolation=T.InterpolationMode.BILINEAR))
    t_ops.append(T.ToTensor())  # [0,1]
    t_lpips = T.Compose(t_ops)

    ds = BAPPS2AFCDataset(subset_dir, transform_lpips=t_lpips)

    def collate_fn(batch):
        refs = torch.stack([b["ref_lpips"] for b in batch])
        p0s = torch.stack([b["p0_lpips"] for b in batch])
        p1s = torch.stack([b["p1_lpips"] for b in batch])
        judges = torch.tensor([b["judge"] for b in batch], dtype=torch.float32)
        return refs, p0s, p1s, judges

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    lpips_fn = lpips_lib.LPIPS(net=lpips_net).to(device)
    lpips_fn.eval()

    total = 0.0
    count = 0
    with torch.no_grad():
        for refs, p0s, p1s, judges in dl:
            refs = refs.to(device)
            p0s = p0s.to(device)
            p1s = p1s.to(device)

            d0 = lpips_fn(refs, p0s, normalize=True).view(-1)
            d1 = lpips_fn(refs, p1s, normalize=True).view(-1)

            choose_p1 = (d1 < d0).float().cpu()
            judges = judges.cpu()

            scores = choose_p1 * judges + (1.0 - choose_p1) * (1.0 - judges)
            total += scores.sum().item()
            count += scores.numel()

    return float(total / max(count, 1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS and SimCLR on BAPPS 2AFC")
    parser.add_argument("--bapps-root", type=str, required=True,
                        help="Path to BAPPS 2AFC root, e.g., /path/PerceptualSimilarity/dataset/2afc")
    parser.add_argument("--subsets", nargs="+", default=[
        "val/traditional", "val/cnn", "val/superres", "val/deblur", "val/color", "val/frameinterp"
    ], help="Subsets to evaluate (relative to --bapps-root)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Force device (default: auto)")

    # LPIPS options
    parser.add_argument("--eval-lpips", action="store_true", help="Evaluate LPIPS")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone")
    parser.add_argument("--lpips-size", type=int, default=256,
                        help="Resize HxW for LPIPS batching (e.g., 256).")

    # SimCLR options
    parser.add_argument("--eval-simclr", action="store_true", help="Evaluate SimCLR embeddings")
    parser.add_argument("--simclr-checkpoint", type=str, default=None, help="Path to SimCLR checkpoint")
    parser.add_argument("--simclr-config", type=str, default=None, help="Path to SimCLR config YAML")
    parser.add_argument("--simclr-input-size", type=int, default=224, help="Resize for SimCLR backbone input")
    parser.add_argument("--simclr-distance", type=str, default="cosine", choices=["cosine", "euclidean"],
                        help="Distance metric for SimCLR features.")
    parser.add_argument("--no-simclr-normalize", action="store_true",
                        help="Disable L2-normalization of SimCLR features before distance.")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    results: Dict[str, Dict[str, float]] = {}

    # Optionally load SimCLR
    simclr_model = None
    if args.eval_simclr:
        assert args.simclr_checkpoint is not None and args.simclr_config is not None, \
            "--simclr-checkpoint and --simclr-config are required when --eval-simclr is set"
        simclr_model = load_simclr_model(args.simclr_checkpoint, args.simclr_config, device)

    all_lpips = []
    all_simclr = []

    for subset in args.subsets:
        subset_dir = os.path.join(args.bapps_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"[WARN] Skipping missing subset: {subset_dir}")
            continue

        res = {}
        if args.eval_lpips:
            acc_lpips = evaluate_lpips_2afc(
                subset_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lpips_net=args.lpips_net,
                lpips_size=args.lpips_size,
            )
            res["lpips"] = acc_lpips
            all_lpips.append(acc_lpips)
            print(f"LPIPS[{args.lpips_net}]  {subset:>16}: {acc_lpips*100:.2f}%")

        if args.eval_simclr and simclr_model is not None:
            acc_simclr = evaluate_simclr_2afc(
                subset_dir,
                model=simclr_model,
                device=device,
                input_size=args.simclr_input_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                distance=args.simclr_distance,
                normalize=(not args.no_simclr_normalize),
            )
            tag = f"simclr_{args.simclr_distance}{'_unnorm' if args.no_simclr_normalize else ''}"
            res[tag] = acc_simclr
            all_simclr.append(acc_simclr)
            print(f"SimCLR[{args.simclr_distance}{', unnorm' if args.no_simclr_normalize else ''}] {subset:>16}: {acc_simclr*100:.2f}%")

        results[subset] = res

    # Overall averages
    if all_lpips:
        print(f"LPIPS[{args.lpips_net}]  OVERALL: {np.mean(all_lpips)*100:.2f}%")
    if all_simclr:
        print(f"SimCLR[{args.simclr_distance}{', unnorm' if args.no_simclr_normalize else ''}] OVERALL: {np.mean(all_simclr)*100:.2f}%")


if __name__ == "__main__":
    main()
