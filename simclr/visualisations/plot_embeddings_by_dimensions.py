import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

def load_embeddings_and_prepare_df(embeddings_path, dim1, dim2):
    """Load embeddings from CSV file and prepare DataFrame"""
    df = pd.read_csv(embeddings_path)
    
    # First column should be filename
    if df.columns[0].lower() in ['filename', 'file', 'name', 'image']:
        filenames = df.iloc[:, 0].values
        embeddings_df = df.iloc[:, 1:]  # All embedding columns
    else:
        raise ValueError("First column should be 'filename' or similar identifier")
    
    # Convert 1-based indices to 0-based for internal use
    dim1_idx = dim1 - 1
    dim2_idx = dim2 - 1
    
    # Check if specified dimensions exist
    if dim1_idx < 0 or dim1_idx >= len(embeddings_df.columns) or dim2_idx < 0 or dim2_idx >= len(embeddings_df.columns):
        raise ValueError(f"Dimension indices out of range. Available dimensions: 1-{len(embeddings_df.columns)} (you provided {dim1}, {dim2})")
    
    # Get dimension names
    dim1_name = embeddings_df.columns[dim1_idx]
    dim2_name = embeddings_df.columns[dim2_idx]
    
    print(f"Dimension 1: {dim1_name} (user input: {dim1})")
    print(f"Dimension 2: {dim2_name} (user input: {dim2})")
    
    # Create a new DataFrame with the required columns
    plot_df = pd.DataFrame({
        'image_name': filenames,
        'x_coord': embeddings_df.iloc[:, dim1_idx].values,
        'y_coord': embeddings_df.iloc[:, dim2_idx].values
    })
    
    return plot_df, dim1_name, dim2_name

def find_image_path(filename, data_dir):
    """Find the full path to an image file in the data directory"""
    # Common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Try the filename as-is first
    for ext in [''] + extensions:
        # Remove existing extension if trying a new one
        base_name = os.path.splitext(filename)[0] if ext else filename
        full_path = os.path.join(data_dir, base_name + ext)
        if os.path.exists(full_path):
            return full_path
    
    # If not found, try searching recursively
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[0] == os.path.splitext(filename)[0]:
                return os.path.join(root, file)
    
    return None

def create_thumbnail_with_aspect_ratio(image_path, max_size, transparent_bg=True):
    """Create a thumbnail that preserves aspect ratio and optionally has transparent background"""
    try:
        # Open image with PIL
        img = Image.open(image_path)
        
        # Convert to RGBA for transparency support
        if transparent_bg and img.mode != 'RGBA':
            img = img.convert('RGBA')
        elif not transparent_bg and img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate new size preserving aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        if transparent_bg:
            # Create transparent background
            new_img = Image.new('RGBA', (max_size, max_size), (255, 255, 255, 0))
            # Center the image
            x_offset = (max_size - img.width) // 2
            y_offset = (max_size - img.height) // 2
            new_img.paste(img, (x_offset, y_offset), img if img.mode == 'RGBA' else None)
            return np.array(new_img)
        else:
            # Create white background
            new_img = Image.new('RGB', (max_size, max_size), (255, 255, 255))
            # Center the image
            x_offset = (max_size - img.width) // 2
            y_offset = (max_size - img.height) // 2
            new_img.paste(img, (x_offset, y_offset))
            return np.array(new_img)
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_thumbnail(args):
    """Load and create thumbnail with preserved aspect ratio"""
    image_path, max_size, transparent_bg = args
    return create_thumbnail_with_aspect_ratio(image_path, max_size, transparent_bg)

def plot_embeddings_with_aspect_preserved_images(plot_df, data_dir, output_path, dim1_name, dim2_name,
                                                thumb_size=80, transparent_bg=True, figsize=(16, 12), 
                                                dpi=150, normalize=False):
    """
    Create scatter plot with aspect-ratio preserved thumbnails
    """
    print("Creating visualization with aspect-ratio preserved thumbnails...")
    
    # Normalize coordinates if requested
    if normalize:
        x_coords = (plot_df['x_coord'] - plot_df['x_coord'].min()) / (plot_df['x_coord'].max() - plot_df['x_coord'].min())
        y_coords = (plot_df['y_coord'] - plot_df['y_coord'].min()) / (plot_df['y_coord'].max() - plot_df['y_coord'].min())
    else:
        x_coords = plot_df['x_coord'].values
        y_coords = plot_df['y_coord'].values
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Find valid image paths
    print("Finding image files...")
    valid_data = []
    for idx, row in plot_df.iterrows():
        img_path = find_image_path(row['image_name'], data_dir)
        if img_path:
            valid_data.append((img_path, x_coords[idx], y_coords[idx], row['image_name']))
        else:
            print(f"Warning: Could not find image for {row['image_name']}")
    
    if not valid_data:
        raise ValueError("No valid images found. Check your data directory and filenames.")
    
    print(f"Found {len(valid_data)} valid images out of {len(plot_df)} total")
    
    # Load thumbnails in parallel
    print("Loading and processing thumbnails...")
    image_args = [(img_path, thumb_size, transparent_bg) for img_path, _, _, _ in valid_data]
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        thumbnails = list(tqdm(
            executor.map(get_thumbnail, image_args),
            total=len(image_args),
            desc="Creating thumbnails"
        ))
    
    # Plot thumbnails
    print("Placing images on plot...")
    placed_count = 0
    for i, ((img_path, x, y, filename), thumbnail) in enumerate(zip(valid_data, thumbnails)):
        if thumbnail is not None:
            # Create AnnotationBbox with the thumbnail
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(thumbnail, zoom=1.0),
                (x, y),
                frameon=False,
                pad=0.0,
            )
            ax.add_artist(imagebox)
            placed_count += 1
    
    print(f"Successfully placed {placed_count} images")
    
    # Set labels and title
    title = f'Bird Images by Embedding Dimensions ({dim1_name} vs {dim2_name})'
    if normalize:
        title += ' (normalized)'
    
    ax.set_xlabel(f'Dimension {dim1_name}', fontsize=14)
    ax.set_ylabel(f'Dimension {dim2_name}', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    x_padding = x_range * 0.1
    y_padding = y_range * 0.1
    
    ax.set_xlim(min(x_coords) - x_padding, max(x_coords) + x_padding)
    ax.set_ylim(min(y_coords) - y_padding, max(y_coords) + y_padding)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    print(f"Visualization saved to: {output_path}")
    plt.close()

def main(args):
    print(f"Loading embeddings from {args.embeddings_csv}...")
    plot_df, dim1_name, dim2_name = load_embeddings_and_prepare_df(
        args.embeddings_csv, args.dim1, args.dim2
    )
    print(f"Loaded {len(plot_df)} embeddings")
    print(f"Using dimensions: {dim1_name} (user input: {args.dim1}) vs {dim2_name} (user input: {args.dim2})")
    
    # Limit number of images if specified
    if args.max_images and len(plot_df) > args.max_images:
        plot_df = plot_df.sample(n=args.max_images, random_state=42).reset_index(drop=True)
        print(f"Randomly selected {args.max_images} images for visualization")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Generate and save visualization
    plot_embeddings_with_aspect_preserved_images(
        plot_df=plot_df,
        data_dir=args.data_dir,
        output_path=args.output_path,
        dim1_name=dim1_name,
        dim2_name=dim2_name,
        thumb_size=args.thumb_size,
        transparent_bg=args.transparent_bg,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        normalize=args.normalize
    )

def get_args():
    parser = argparse.ArgumentParser(
        description='Plot bird images with preserved aspect ratios according to specific embedding dimensions'
    )
    parser.add_argument('--embeddings_csv', type=str, required=True,
                       help='Path to the CSV file containing embeddings (first column: filename)')
    parser.add_argument('--dim1', type=int, required=True,
                       help='Index of first embedding dimension to plot (1-based)')
    parser.add_argument('--dim2', type=int, required=True,
                       help='Index of second embedding dimension to plot (1-based)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the bird images')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the visualization (e.g., plot.png)')
    parser.add_argument('--thumb_size', type=int, default=80,
                       help='Maximum size of thumbnail images in pixels (default: 80)')
    parser.add_argument('--max_images', type=int, default=1000,
                       help='Maximum number of images to plot (default: 1000)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[16, 12],
                       help='Figure size as width height (default: 16 12)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for the output image (default: 300)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize the embedding dimensions to [0,1] range')
    parser.add_argument('--transparent_bg', action='store_true', default=True,
                       help='Use transparent background for thumbnails (default: True)')
    parser.add_argument('--no_transparent_bg', dest='transparent_bg', action='store_false',
                       help='Use white background instead of transparent')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
