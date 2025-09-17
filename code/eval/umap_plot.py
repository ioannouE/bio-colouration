import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import re


def get_thumbnail(path, size=(100, 100)):
    """Load and resize an image to create a thumbnail"""
    try:
        img = Image.open(path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return np.asarray(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def imscatter(x, y, thumbnails, ax=None, zoom=0.3, pad=0.0):
    """Plot images as scatter points on the given axes"""
    if ax is None:
        ax = plt.gca()
    for xi, yi, img in zip(x, y, thumbnails):
        try:
            imagebox = OffsetImage(img, zoom=zoom, resample=True)
            # Set box_alignment to (0.5, 0.5) to ensure the image is centered at the data point
            ab = AnnotationBbox(imagebox, (xi, yi), frameon=False, 
                               box_alignment=(0.5, 0.5), pad=pad)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Could not display image at ({xi}, {yi}): {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process UMAP embeddings.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the embeddings CSV file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for the result CSV')
    parser.add_argument('--subset_size', type=int, default=5000, help='Size of the random subset to use')
    return parser.parse_args()


def compute_output_path(csv_path, output_dir, subset_size):
    """Compute the output file path based on the CSV filename pattern"""
    match = re.search(r'embeddings_config_kornia02_(\d{8}T\d{4})\.csv$', csv_path)
    if match:
        timestamp = match.group(1)
        output_file_name = f"{timestamp}_umap_{subset_size}.png"
        return os.path.join(output_dir, output_file_name)
    else:
        raise ValueError('CSV path does not match expected pattern.')


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Compute output file path
    output_path = compute_output_path(args.csv_path, args.output_path, args.subset_size)
    
    # === Step 1: Load embeddings CSV ===
    print(f"Loading embeddings from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # === Select a random subset ===
    print(f"Selecting random subset of {args.subset_size} samples")
    df_subset = df.sample(n=args.subset_size, random_state=42)
    
    filenames = df_subset["filename"].values
    X = df_subset.drop(columns=["filename"]).values.astype(np.float32)
    
    # === Step 2: UMAP Embedding ===
    print("Computing UMAP embedding...")
    embedding_2d = umap.UMAP(n_components=2, random_state=42, metric='cosine', min_dist=0.4).fit_transform(X)
    print("UMAP Embedding shape:", embedding_2d.shape)
    
    # === Step 3: Plot with image thumbnails ===
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Construct full image paths
    image_paths = [os.path.join(args.image_dir, fname) for fname in filenames]
    
    # Load images in parallel
    print("Loading and resizing images...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        thumbnails = list(tqdm(
            executor.map(get_thumbnail, image_paths),
            total=len(image_paths),
            desc="Loading images"
        ))
    
    # Use the same embedding coordinates for both scatter and imscatter
    x_coords = embedding_2d[:, 0]
    y_coords = embedding_2d[:, 1]
    
    # Debug: Print sample coordinates
    print("Sample x coordinates:", x_coords[:5])
    print("Sample y coordinates:", y_coords[:5])
    
    # Plot with scatter
    plt.scatter(x_coords, y_coords, s=10, alpha=0.7)
    
    # Plot with imscatter
    imscatter(x_coords, y_coords, thumbnails, ax=ax, zoom=0.4, pad=0.02)
    
    ax.set_title("UMAP with Image Thumbnails", fontsize=18)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    print(f"Saving figure to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    main()
