import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.offsetbox as offsetbox
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_embeddings_from_csv(embeddings_path):
    """Load embeddings from CSV file"""
    df = pd.read_csv(embeddings_path)
    
    # Separate filenames (if present) from embeddings
    if df.columns[0].lower() in ['filename', 'file', 'name', 'image']:
        filenames = df.iloc[:, 0].values
        embeddings = df.iloc[:, 1:].values
    else:
        filenames = None
        embeddings = df.values
    
    return embeddings, filenames

def get_thumbnail(args):
    """Load and resize an image to create a thumbnail"""
    path, size = args
    try:
        img = Image.open(path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return np.asarray(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def plot_embeddings_with_images(embeddings, filenames, data_dir, output_path, method='umap', 
                              thumb_size=100, max_images=1000, random_state=42, grid_scale=1.0):
    """
    Create and save a 2D visualization using thumbnails of the actual bird images
    
    Parameters:
        ...
        grid_scale: float, controls the density of images (lower values = more dense, higher = more sparse)
    """
    print(f"Reducing dimensionality using {method.upper()}...")
    
    # Scale the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Reduce dimensionality
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings_scaled)
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings_scaled)
    
    # If we have too many images, randomly sample them
    if len(filenames) > max_images:
        print(f"Sampling {max_images} images from {len(filenames)} total images...")
        indices = np.random.choice(len(filenames), max_images, replace=False)
        embedding_2d = embedding_2d[indices]
        filenames = filenames[indices]
    
    # Prepare image paths for parallel loading
    image_paths = [(os.path.join(data_dir, fname), (thumb_size, thumb_size)) 
                  for fname in filenames]
    
    # Load images in parallel
    print("Loading and resizing images...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        thumbnails = list(tqdm(
            executor.map(get_thumbnail, image_paths),
            total=len(image_paths),
            desc="Loading images"
        ))
    
    # Filter out None values (failed loads) and their corresponding embeddings
    valid_indices = [i for i, thumb in enumerate(thumbnails) if thumb is not None]
    thumbnails = [thumbnails[i] for i in valid_indices]
    embedding_2d = embedding_2d[valid_indices]
    
    print("Creating visualization...")
    # Create the figure
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
    
    # Calculate figure bounds
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Create grid for overlap detection
    grid_size = (thumb_size / 100) * grid_scale  # Adjustable grid size
    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)
    occupied_grid = np.zeros((len(x_grid)-1, len(y_grid)-1), dtype=bool)
    
    # Plot thumbnails with grid-based overlap detection
    print("Placing images on plot...")
    for idx, (x0, y0) in enumerate(tqdm(embedding_2d, desc="Plotting images")):
        # Convert coordinates to grid indices
        x_idx = np.searchsorted(x_grid, x0) - 1
        y_idx = np.searchsorted(y_grid, y0) - 1
        
        # Check if grid cell is occupied
        if x_idx >= 0 and y_idx >= 0 and x_idx < occupied_grid.shape[0] and y_idx < occupied_grid.shape[1]:
            if occupied_grid[x_idx, y_idx]:
                continue
            
            # Mark grid cell as occupied
            occupied_grid[x_idx, y_idx] = True
            
            # Add thumbnail to plot
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(thumbnails[idx], zoom=0.5),
                (x0, y0),
                frameon=False,
                pad=0.0,
            )
            ax.add_artist(imagebox)
    
    plt.title(f'2D {method.upper()} projection with bird thumbnails')
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    plt.grid(True, alpha=0.3)  # Optional: add a light grid
    
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Saved. --- ")

def main(args):
    print(f"Loading embeddings from {args.embeddings_csv}...")
    embeddings, filenames = load_embeddings_from_csv(args.embeddings_csv)
    print(f"Loaded {len(embeddings)} embeddings")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate and save UMAP visualization
    umap_output_path = os.path.join(args.output_dir, 'umap_visualization_with_images.png')
    plot_embeddings_with_images(
        embeddings, filenames, args.data_dir, umap_output_path, 
        method='umap', thumb_size=args.thumb_size, max_images=args.max_images,
        grid_scale=args.grid_scale
    )
    
    # Generate and save t-SNE visualization
    tsne_output_path = os.path.join(args.output_dir, 'tsne_visualization_with_images.png')
    plot_embeddings_with_images(
        embeddings, filenames, args.data_dir, tsne_output_path, 
        method='tsne', thumb_size=args.thumb_size, max_images=args.max_images,
        grid_scale=args.grid_scale
    )

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Generate UMAP and t-SNE visualizations with bird thumbnails')
    parser.add_argument('--embeddings_csv', type=str, required=True,
                      help='Path to the CSV file containing embeddings')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the bird images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the visualizations')
    parser.add_argument('--thumb_size', type=int, default=100,
                      help='Size of thumbnail images in pixels')
    parser.add_argument('--max_images', type=int, default=600,
                      help='Maximum number of images to plot')
    parser.add_argument('--grid_scale', type=float, default=1.0,
                      help='Scale factor for grid size (lower = more dense images)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)