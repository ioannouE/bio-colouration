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

# Local SimCLR model (keep consistent with generate_embeddings.py and eval_bapps_2afc.py)
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


class BAPPSJNDDataset(Dataset):
    """BAPPS JND pair dataset (p0, p1, same).

    Expects directory structure like:
      subset_dir/
        p0/
        p1/
        same/
    Matches by basename across the three folders (robust to count/order mismatches).
    """

    def __init__(self, subset_dir: str, transform_lpips: Optional[T.Compose] = None,
                 transform_simclr: Optional[T.Compose] = None,
                 p0_dir: str = "p0", p1_dir: str = "p1", same_dir: str = "same",
                 same_exts: Tuple[str, ...] = ("npy",)):
        self.subset_dir = subset_dir
        self.dir_p0 = os.path.join(subset_dir, p0_dir)
        self.dir_p1 = os.path.join(subset_dir, p1_dir)
        self.dir_same = os.path.join(subset_dir, same_dir)

        if not os.path.isdir(self.dir_p0) or not os.path.isdir(self.dir_p1) or not os.path.isdir(self.dir_same):
            raise FileNotFoundError(
                f"Expected '{p0_dir}/', '{p1_dir}/', and '{same_dir}/' under {subset_dir}")

        def list_files(d: str, exts: Tuple[str, ...]) -> List[str]:
            paths: List[str] = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(d, f"*.{ext}")))
            return sorted(paths)

        IMG_EXTS: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp")
        p0_files = list_files(self.dir_p0, IMG_EXTS)
        p1_files = list_files(self.dir_p1, IMG_EXTS)

        p0_map = {os.path.splitext(os.path.basename(p))[0]: p for p in p0_files}
        p1_map = {os.path.splitext(os.path.basename(p))[0]: p for p in p1_files}

        same_map: Dict[str, str] = {}
        for ext in same_exts:
            for p in glob.glob(os.path.join(self.dir_same, f"*.{ext}")):
                b = os.path.splitext(os.path.basename(p))[0]
                if b not in same_map:
                    same_map[b] = p

        common = sorted(list(p0_map.keys() & p1_map.keys() & same_map.keys()))
        if len(common) == 0:
            raise RuntimeError(
                f"No matching (p0,p1,same) triplets found in {subset_dir}. Check filenames and extensions.")

        self.triplets: List[Tuple[str, str, str]] = [(p0_map[k], p1_map[k], same_map[k]) for k in common]

        self.transform_lpips = transform_lpips
        self.transform_simclr = transform_simclr

    def __len__(self) -> int:
        return len(self.triplets)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _load_label(self, path: str) -> float:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".npy":
                arr = np.load(path, allow_pickle=False)
            elif ext == ".npz":
                arrz = np.load(path, allow_pickle=False)
                if "arr_0" in arrz:
                    arr = arrz["arr_0"]
                else:
                    # take the first array
                    key0 = list(arrz.keys())[0]
                    arr = arrz[key0]
            elif ext in (".txt", ".csv"):
                arr = np.loadtxt(path, dtype=np.float32)
            else:
                arr = np.load(path, allow_pickle=False)
        except Exception:
            arr = np.loadtxt(path, dtype=np.float32)
        return float(np.array(arr, dtype=np.float32).reshape(-1)[0])

    def __getitem__(self, idx: int):
        p0_path, p1_path, same_path = self.triplets[idx]
        p0 = self._load_image(p0_path)
        p1 = self._load_image(p1_path)
        # 'same' is a fraction in [0,1] (or {0,1}); higher means more likely 'same'
        judge = self._load_label(same_path)

        sample = {
            "judge": judge,
        }

        if self.transform_lpips is not None:
            sample["p0_lpips"] = self.transform_lpips(p0)
            sample["p1_lpips"] = self.transform_lpips(p1)

        if self.transform_simclr is not None:
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
    # Returns optional L2-normalized backbone features
    with torch.no_grad():
        feats = model.backbone(x).flatten(1)
        if normalize:
            feats = F.normalize(feats, dim=1)
    return feats


# --- Metrics: ROC-AUC and best-threshold accuracy ---

def _rankdata_average_ties(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    sorted_scores = scores[order]
    n = len(scores)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average_ties(y_score)
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_accuracy_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    # Predict positive if score >= thresh
    # Evaluate accuracy over midpoints between sorted unique scores
    order = np.argsort(y_score)
    scores_sorted = y_score[order]
    y_sorted = y_true[order]
    # Candidate thresholds: midpoints between consecutive scores, plus -inf and +inf edges
    uniq = np.unique(scores_sorted)
    if len(uniq) == 1:
        # any threshold yields the same prediction; choose that value
        thresh = float(uniq[0])
        acc = max(y_true.mean(), 1 - y_true.mean())
        return thresh, float(acc)
    mids = (uniq[:-1] + uniq[1:]) * 0.5
    candidates = np.concatenate(([-np.inf], mids, [np.inf]))

    best_acc = -1.0
    best_t = 0.0
    n = len(y_true)
    for t in candidates:
        pred = (y_score >= t).astype(np.int32)
        acc = (pred == y_true).sum() / n
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t), float(best_acc)


def evaluate_simclr_jnd(subset_dir: str, model: SimCLRModel, device: torch.device,
                        input_size: int, batch_size: int, num_workers: int,
                        distance: str = "cosine", normalize: bool = True,
                        positive_label: str = "different", judge_threshold: float = 0.5,
                        p0_dir: str = "p0", p1_dir: str = "p1", same_dir: str = "same",
                        same_exts: Optional[List[str]] = None) -> Dict[str, float]:
    t_simclr = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if same_exts is None:
        same_exts = ["npy"]
    ds = BAPPSJNDDataset(
        subset_dir,
        transform_simclr=t_simclr,
        p0_dir=p0_dir, p1_dir=p1_dir, same_dir=same_dir,
        same_exts=tuple(same_exts),
    )

    def collate_fn(batch):
        p0s = torch.stack([b["p0_simclr"] for b in batch])
        p1s = torch.stack([b["p1_simclr"] for b in batch])
        judges = torch.tensor([b["judge"] for b in batch], dtype=torch.float32)
        return p0s, p1s, judges

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for p0s, p1s, judges in dl:
        p0s = p0s.to(device)
        p1s = p1s.to(device)

        z_p0 = simclr_feature_extractor(model, p0s, normalize=normalize)
        z_p1 = simclr_feature_extractor(model, p1s, normalize=normalize)

        if distance == "cosine":
            d = 1.0 - F.cosine_similarity(z_p0, z_p1, dim=1)
        elif distance == "euclidean":
            d = F.pairwise_distance(z_p0, z_p1, p=2)
        else:
            raise ValueError(f"Unknown distance: {distance}")

        all_scores.append(d.detach().cpu().numpy())
        all_labels.append(judges.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # labels represent 'same' probability/ratio; higher means 'same'
    if positive_label == "different":
        y_true = (labels < judge_threshold).astype(np.int32)
    elif positive_label == "same":
        y_true = (labels >= judge_threshold).astype(np.int32)
    else:
        raise ValueError("positive_label must be 'different' or 'same'")

    auc = roc_auc_score_binary(y_true, scores)
    best_t, best_acc = best_accuracy_threshold(y_true, scores)

    return {"roc_auc": float(auc), "acc_best_t": float(best_acc), "best_thresh": float(best_t)}


def evaluate_lpips_jnd(subset_dir: str, device: torch.device, batch_size: int, num_workers: int,
                       lpips_net: str = "alex", lpips_size: int = 256,
                       positive_label: str = "different", judge_threshold: float = 0.5,
                       p0_dir: str = "p0", p1_dir: str = "p1", same_dir: str = "same",
                       same_exts: Optional[List[str]] = None) -> Dict[str, float]:
    if lpips_lib is None:
        raise RuntimeError("lpips is not installed. Please `pip install lpips`. ")

    # LPIPS expects inputs in [0,1] if normalize=True is passed to forward
    t_ops = []
    if lpips_size and lpips_size > 0:
        t_ops.append(T.Resize((lpips_size, lpips_size), interpolation=T.InterpolationMode.BILINEAR))
    t_ops.append(T.ToTensor())  # [0,1]
    t_lpips = T.Compose(t_ops)

    if same_exts is None:
        same_exts = ["npy"]
    ds = BAPPSJNDDataset(
        subset_dir,
        transform_lpips=t_lpips,
        p0_dir=p0_dir, p1_dir=p1_dir, same_dir=same_dir,
        same_exts=tuple(same_exts),
    )

    def collate_fn(batch):
        p0s = torch.stack([b["p0_lpips"] for b in batch])
        p1s = torch.stack([b["p1_lpips"] for b in batch])
        judges = torch.tensor([b["judge"] for b in batch], dtype=torch.float32)
        return p0s, p1s, judges

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    lpips_fn = lpips_lib.LPIPS(net=lpips_net).to(device)
    lpips_fn.eval()

    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for p0s, p1s, judges in dl:
            p0s = p0s.to(device)
            p1s = p1s.to(device)

            d = lpips_fn(p0s, p1s, normalize=True).view(-1)
            all_scores.append(d.detach().cpu().numpy())
            all_labels.append(judges.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # labels represent 'same' probability/ratio; higher means 'same'
    if positive_label == "different":
        y_true = (labels < judge_threshold).astype(np.int32)
    elif positive_label == "same":
        y_true = (labels >= judge_threshold).astype(np.int32)
    else:
        raise ValueError("positive_label must be 'different' or 'same'")

    auc = roc_auc_score_binary(y_true, scores)
    best_t, best_acc = best_accuracy_threshold(y_true, scores)

    return {"roc_auc": float(auc), "acc_best_t": float(best_acc), "best_thresh": float(best_t)}


def _auto_discover_subsets(root: str) -> List[str]:
    # If root itself contains p0/p1/same, treat as single subset "."
    if all(os.path.isdir(os.path.join(root, x)) for x in ["p0", "p1", "same"]):
        return ["."]
    # Else, use immediate subdirectories
    subs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    lvl1 = [d for d in subs if all(os.path.isdir(os.path.join(root, d, x)) for x in ["p0", "p1", "same"])]
    if lvl1:
        return lvl1

    # Case 3: nested one more level (e.g., val/cnn, val/traditional)
    lvl2: List[str] = []
    for d in subs:
        d_path = os.path.join(root, d)
        subsubs = [s for s in sorted(os.listdir(d_path)) if os.path.isdir(os.path.join(d_path, s))]
        for s in subsubs:
            if all(os.path.isdir(os.path.join(d_path, s, x)) for x in ["p0", "p1", "same"]):
                lvl2.append(os.path.join(d, s))
    if lvl2:
        return lvl2

    # Fallback: return immediate subdirectories
    return subs


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS and SimCLR on BAPPS JND")
    parser.add_argument("--bapps-root", type=str, required=True,
                        help="Path to BAPPS JND root, e.g., /path/PerceptualSimilarity/dataset/jnd")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="Subsets to evaluate (relative to --bapps-root). If omitted, auto-discover.")
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

    # JND label handling
    parser.add_argument("--positive-label", type=str, default="different", choices=["different", "same"],
                        help="Which class is considered positive for metrics")
    parser.add_argument("--judge-threshold", type=float, default=0.5,
                        help="Threshold on judge in [0,1] to binarize labels")

    # JND directory names and label extensions
    parser.add_argument("--jnd-p0-dir", type=str, default="p0", help="Folder name for first image in pair")
    parser.add_argument("--jnd-p1-dir", type=str, default="p1", help="Folder name for second image in pair")
    parser.add_argument("--jnd-same-dir", type=str, default="same", help="Folder name for labels")
    parser.add_argument("--jnd-same-exts", type=str, nargs="+", default=["npy"],
                        help="Allowed label file extensions (without dots), e.g., npy npz txt")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Optionally load SimCLR
    simclr_model = None
    if args.eval_simclr:
        assert args.simclr_checkpoint is not None and args.simclr_config is not None, \
            "--simclr-checkpoint and --simclr-config are required when --eval-simclr is set"
        simclr_model = load_simclr_model(args.simclr_checkpoint, args.simclr_config, device)

    # Determine subsets
    subsets = args.subsets if args.subsets is not None else _auto_discover_subsets(args.bapps_root)
    if not subsets:
        print(f"[WARN] No subsets found under {args.bapps_root}")
        return

    results: Dict[str, Dict[str, float]] = {}
    all_lpips_auc: List[float] = []
    all_simclr_auc: List[float] = []

    for subset in subsets:
        subset_dir = os.path.join(args.bapps_root, subset)
        if subset == ".":
            subset_dir = args.bapps_root
        if not os.path.isdir(subset_dir):
            print(f"[WARN] Skipping missing subset: {subset_dir}")
            continue

        res = {}
        if args.eval_lpips:
            m = evaluate_lpips_jnd(
                subset_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lpips_net=args.lpips_net,
                lpips_size=args.lpips_size,
                positive_label=args.positive_label,
                judge_threshold=args.judge_threshold,
                p0_dir=args.jnd_p0_dir,
                p1_dir=args.jnd_p1_dir,
                same_dir=args.jnd_same_dir,
                same_exts=args.jnd_same_exts,
            )
            res["lpips_roc_auc"] = m["roc_auc"]
            res["lpips_acc_best_t"] = m["acc_best_t"]
            all_lpips_auc.append(m["roc_auc"]) if not np.isnan(m["roc_auc"]) else None
            print(f"LPIPS[{args.lpips_net}]  {subset:>16}: AUC={m['roc_auc']*100:.2f}%, Acc@best={m['acc_best_t']*100:.2f}% (t={m['best_thresh']:.4f})")

        if args.eval_simclr and simclr_model is not None:
            m = evaluate_simclr_jnd(
                subset_dir,
                model=simclr_model,
                device=device,
                input_size=args.simclr_input_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                distance=args.simclr_distance,
                normalize=(not args.no_simclr_normalize),
                positive_label=args.positive_label,
                judge_threshold=args.judge_threshold,
                p0_dir=args.jnd_p0_dir,
                p1_dir=args.jnd_p1_dir,
                same_dir=args.jnd_same_dir,
                same_exts=args.jnd_same_exts,
            )
            tag_prefix = f"simclr_{args.simclr_distance}{'_unnorm' if args.no_simclr_normalize else ''}"
            res[f"{tag_prefix}_roc_auc"] = m["roc_auc"]
            res[f"{tag_prefix}_acc_best_t"] = m["acc_best_t"]
            all_simclr_auc.append(m["roc_auc"]) if not np.isnan(m["roc_auc"]) else None
            print(f"SimCLR[{args.simclr_distance}{', unnorm' if args.no_simclr_normalize else ''}] {subset:>16}: AUC={m['roc_auc']*100:.2f}%, Acc@best={m['acc_best_t']*100:.2f}% (t={m['best_thresh']:.4f})")

        results[subset] = res

    # Overall averages
    if all_lpips_auc:
        print(f"LPIPS[{args.lpips_net}]  OVERALL AUC: {np.mean(all_lpips_auc)*100:.2f}%")
    if all_simclr_auc:
        print(f"SimCLR[{args.simclr_distance}{', unnorm' if args.no_simclr_normalize else ''}] OVERALL AUC: {np.mean(all_simclr_auc)*100:.2f}%")


if __name__ == "__main__":
    main()
