import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def load_embeddings_from_csv(embeddings_path):
    """Load embeddings from CSV file"""
    # Read the CSV file
    df = pd.read_csv(embeddings_path)
    
    # Separate filenames (if present) from embeddings
    # Assuming first column might be filenames/index
    if df.columns[0].lower() in ['filename', 'file', 'name', 'image']:
        filenames = df.iloc[:, 0].values
        embeddings = df.iloc[:, 1:].values
    else:
        filenames = None
        embeddings = df.values
    
    return embeddings, filenames

def plot_embeddings_reduction(embeddings, output_path, method='umap', random_state=42):
    """
    Create and save a 2D visualization of the embeddings using UMAP or t-SNE
    
    Parameters:
    -----------
    embeddings : np.ndarray
        The high-dimensional embeddings to reduce
    output_path : str
        Path to save the visualization
    method : str
        'umap' or 'tsne' for the reduction method
    random_state : int
        Random seed for reproducibility
    """
    plt.figure(figsize=(12, 10))
    
    # Scale the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings_scaled)
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state)
        embedding_2d = reducer.fit_transform(embeddings_scaled)
    
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6)
    plt.title(f'2D {method.upper()} projection of the embeddings')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load embeddings from CSV
    print(f"Loading embeddings from: {args.embeddings_csv}")
    embeddings, filenames = load_embeddings_from_csv(args.embeddings_csv)
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Generate and save UMAP visualization
    umap_output_path = os.path.join(args.output_dir, 'umap_visualization.png')
    print("Generating UMAP visualization...")
    plot_embeddings_reduction(embeddings, umap_output_path, method='umap')
    print(f"UMAP visualization saved to: {umap_output_path}")
    
    # Generate and save t-SNE visualization
    tsne_output_path = os.path.join(args.output_dir, 'tsne_visualization.png')
    print("Generating t-SNE visualization...")
    plot_embeddings_reduction(embeddings, tsne_output_path, method='tsne')
    print(f"t-SNE visualization saved to: {tsne_output_path}")

def get_args():
    parser = argparse.ArgumentParser(description='Generate UMAP and t-SNE visualizations from embeddings CSV')
    parser.add_argument('--embeddings_csv', type=str, required=True,
                        help='Path to the CSV file containing embeddings')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the visualizations')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)