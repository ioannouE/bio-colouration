import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the correct dataset class for multispectral data
from simclr.scripts.multispectral_data import MultispectralTifDataset

# Define the identity transform at the top level to make it picklable by multiprocessing
def identity_transform(x):
    """Returns the input unchanged."""
    return x

def calculate_channel_stats(dataloader, num_channels):
    """Calculates mean and variance for each channel in the dataset."""
    # Welford's algorithm for stable online variance calculation
    count = 0
    mean = np.zeros(num_channels)
    m2 = np.zeros(num_channels)

    for images, filenames in tqdm(dataloader, desc="Calculating Stats"):
        if images is None:
            print(f"Skipping corrupt file: {filenames}")
            continue
        
        # images are (batch, channels, height, width)
        pixels = images.permute(0, 2, 3, 1).reshape(-1, num_channels).numpy()

        for x in pixels:
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

    variance = m2 / (count - 1) if count > 1 else np.zeros(num_channels)
    return mean, variance

def main():
    parser = argparse.ArgumentParser(description='Evaluate channel variance for multispectral data.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the root of the multispectral dataset.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the directory to save outputs.')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top channels to select based on variance.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for the DataLoader.')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # --- 1. Load Dataset ---
    # Provide an identity transform to prevent UnboundLocalError in the dataset class
    # when no transform is specified.
    dataset = MultispectralTifDataset(input_dir=args.dataset_path, transform=identity_transform)
    if len(dataset) == 0:
        print(f"Error: No .tif files found in {args.dataset_path}")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # --- 2. Get Number of Channels ---
    first_image, _ = next(iter(dataloader))
    if first_image is None:
        print("Error: Could not load the first image to determine channel count.")
        return
    num_channels = first_image.shape[1]
    print(f"Detected {num_channels} channels.")

    # --- 3. Calculate Mean and Variance ---
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    channel_means, channel_variances = calculate_channel_stats(dataloader, num_channels)

    # --- 4. Save Full Statistics ---
    stats_df = pd.DataFrame({
        'channel': range(num_channels),
        'mean': channel_means,
        'variance': channel_variances
    })
    stats_df = stats_df.sort_values(by='variance', ascending=False)
    full_stats_path = os.path.join(args.output_path, 'multispectral_channel_stats.csv')
    stats_df.to_csv(full_stats_path, index=False)
    print(f"Full channel stats saved to {full_stats_path}")

    # --- 5. Save Top N Channels ---
    top_n_df = stats_df.head(args.top_n)
    top_n_path = os.path.join(args.output_path, f'top_{args.top_n}_multispectral_channels.csv')
    top_n_df.to_csv(top_n_path, index=False)
    print(f"Top {args.top_n} channels saved to {top_n_path}")
    print("\nTop Channels (by variance):")
    print(top_n_df)

    # --- 6. Generate and Save Plot ---
    plt.figure(figsize=(10, 6))
    plt.bar(stats_df['channel'], stats_df['variance'], color='teal')
    plt.title('Variance per Multispectral Channel')
    plt.xlabel('Channel Index')
    plt.ylabel('Variance')
    plt.xticks(stats_df['channel'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(args.output_path, 'multispectral_channel_variance.png')
    plt.savefig(plot_path)
    print(f"\nVariance plot saved to {plot_path}")

if __name__ == '__main__':
    main()
