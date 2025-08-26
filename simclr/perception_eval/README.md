# Evaluating SimCLR embeddings for Perceptual Judgement

This directory contains code for evaluating SimCLR embeddings for perceptual judgement using the Berkeley-Adobe Perceptual Patch Similarity (BAPPS) dataset. The dataset is introduced in the LPIPS paper (*"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"*). Code for LPIPS and for downloading the BAPPS dataset is available in the [LPIPS repository](https://github.com/richzhang/PerceptualSimilarity).

## Usage

### 2AFC evaluation
BAPPS contains Triplets of small patches: a reference `x`, and two candidates `x0`, `x1`, plus the human answer “which is closer to x?”. The metric “wins” if it picks the same candidate as humans.

- LPIPS distance: `d = LPIPS(x, y)` (lower = more similar).

- SimCLR distance: compute embeddings `f(x)`, `f(y)`; `L2-normalize` then use cosine distance
`d = 1 − cos( f̂(x), f̂(y) )`. (Or use euclidean distance `d = ||f̂(x) - f̂(y)||`)

#### Primary score
2AFC accuracy = `%` triplets where the metric agrees with the human choice.

#### Secondary scores [TODO]
Agreement vs. vote strength (when BAPPS provides human vote proportions): Spearman correlation between the SimCLR scores `|d(x,x1)-d(x,x0)|` and the human vote imbalance.


```bash
python eval_bapps_2afc.py --ckpt_path <path_to_ckpt> --config_path <path_to_config> --subset_dir <path_to_subset_dir> --device <device> --input_size <input_size> --batch_size <batch_size> --num_workers <num_workers> --distance <distance> --normalize <normalize>
```

### Arguments

- `ckpt_path`: Path to the checkpoint of the SimCLR model.
- `config_path`: Path to the config file used for training the SimCLR model.
- `subset_dir`: Path to the BAPPS subset directory.
- `device`: Device to use for evaluation (e.g. 'cuda' or 'cpu').
- `input_size`: Input size for the SimCLR model.
- `batch_size`: Batch size for evaluation.
- `num_workers`: Number of workers for data loading.
- `distance`: Distance metric to use for evaluation (e.g. 'cosine' or 'euclidean').
- `normalize`: Whether to normalize the embeddings before evaluation.

### 2AFC Results

| Dataset        | LPIPS (alex) | SimCLR (cosine, norm) |
| :------------- | ------------:| ---------------------:|
| val/cnn        | 82.72%       | 77.74%                |
| val/traditional| 74.20%       | 73.57%                |
| val/superres   | 70.44%       | 69.70%                |
| val/deblur     | 60.19%       | 58.92%                |
| **val/color**      | **62.46%**       | **62.93%**                |
| val/frameinterp| 62.84%       | 62.84%                |
| **OVERALL**       | **68.81%**       | **67.62%**                |


### JND Evaluation
JND evaluation is based on the JND dataset (of BAPPS) containing pairs of images (one reference and one distorted) and the human answer “are the images the same?”.

```bash
python eval_bapps_jnd.py --ckpt_path <path_to_ckpt> --config_path <path_to_config> --subset_dir <path_to_subset_dir> --device <device> --input_size <input_size> --batch_size <batch_size> --num_workers <num_workers> --distance <distance> --normalize <normalize>
```

### Arguments
JND evaluation arguments:

- `ckpt_path`: Path to the checkpoint of the SimCLR model.
- `config_path`: Path to the config file used for training the SimCLR model.
- `subset_dir`: Path to the BAPPS subset directory.
- `device`: Device to use for evaluation (e.g. 'cuda' or 'cpu').
- `input_size`: Input size for the SimCLR model.
- `batch_size`: Batch size for evaluation.
- `num_workers`: Number of workers for data loading.
- `distance`: Distance metric to use for evaluation (e.g. 'cosine' or 'euclidean').
- `normalize`: Whether to normalize the embeddings before evaluation.

### JND Results

| Dataset        | LPIPS [alex] (AUC) | LPIPS [alex] (Acc@best) | SimCLR [euclidean] (AUC) | SimCLR [euclidean] (Acc@best) |
| :------------- | ------------------:| -----------------:| ---------------------:| ---------------:|
| val/cnn        | 88.78%             | 82.15%            | 81.79%                | 77.46%          |
| val/traditional| 83.21%             | 80.00%            | 87.78%                | 83.85%          |
| **Overall**    | **86.00%**         | –                 | **84.79%**            | –               |
