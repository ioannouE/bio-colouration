import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def get_thumbnail(path, size=(100, 100)):
    """Load and resize an image to create a thumbnail"""
    try:
        img = Image.open(path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return np.asarray(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


# === LOAD IMAGES ===
def load_image(fname, size=(100, 100)):
    try:
        path = os.path.join(image_dir, fname)
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error loading image {fname}: {e}")
        return None




def draw_graph_with_images(G, thumbnails, layout="spring", figsize=(10, 10), zoom=0.4, draw_edges=False):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    layout_func = {
        "spring": nx.spring_layout,
        "spectral": nx.spectral_layout,
        "kamada": nx.kamada_kawai_layout,
        "circular": nx.circular_layout
    }.get(layout, nx.spring_layout)

    pos = layout_func(G)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Draw edges behind thumbnails
    if draw_edges:
        # nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.2, width=0.8)
        weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        
        # Normalize weights for colormap
        norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = cm.plasma  # or 'plasma', 'cool', etc.
        edge_colors = [cmap(norm(w)) for w in weights]

        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=edge_colors,
            edge_cmap=cmap, width=0.4, alpha=0.2
        )


    # Draw images
    for node, coords in pos.items():
        img = thumbnails[node]
        if img is not None:
            imagebox = OffsetImage(img, zoom=zoom, resample=True)
            ab = AnnotationBbox(imagebox, coords, frameon=False, box_alignment=(0.5, 0.5), pad=0.0)
            ax.add_artist(ab)

    # Fix axis limits to fit all nodes
    x_vals, y_vals = zip(*pos.values())
    margin = 0.1
    ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
    ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(True)
    ax.set_title(f"Graph Visualization with Images ({layout} layout)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/outputs/passerines_pagerank_{layout}_{subset_size}_{draw_edges}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()





csv_path = "ViT_embeddings/embeddings_config_kornia02_20250328T1803.csv" 
image_dir = "/Volumes/shared/cooney_lab/Shared/RSE/Image_sets/Passerines-Segmented-Back-RGB-PNG" 

subset_size = 1000
top_n = 16  # number of top-ranked images to visualize

# === LOAD DATA ===
df = pd.read_csv(csv_path)
df_subset = df.sample(n=subset_size, random_state=42)
filenames = df_subset["filename"].values
X = df_subset.drop(columns=["filename"]).values.astype(np.float32)

# === COMPUTE SIMILARITY GRAPH ===
sim_matrix = cosine_similarity(X)
np.fill_diagonal(sim_matrix, 0)
G = nx.from_numpy_array(sim_matrix)


# Construct full image paths
image_paths = [os.path.join(image_dir, fname) for fname in filenames]

# Load images in parallel
print("Loading and resizing images...")
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    thumbnails = list(tqdm(
        executor.map(get_thumbnail, image_paths),
        total=len(image_paths),
        desc="Loading images"
    ))


draw_graph_with_images(G, thumbnails, layout="spring", zoom=0.4, draw_edges=True)



# === RANK WITH PAGERANK ===
pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
top_indices = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:top_n]


top_images = [load_image(filenames[i]) for i in top_indices]
top_scores = [pagerank_scores[i] for i in top_indices]
top_names = [filenames[i] for i in top_indices]

# === DYNAMIC GRID SIZE ===
n_cols = math.ceil(math.sqrt(top_n))
n_rows = math.ceil(top_n / n_cols)

# === DISPLAY AS GRID ===
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
axes = axes.flatten()

for i in range(n_rows * n_cols):
    ax = axes[i]
    if i < len(top_images) and top_images[i] is not None:
        ax.imshow(top_images[i])
        ax.set_title(f"{top_names[i][:20]}...\n{top_scores[i]:.4f}", fontsize=8)
    else:
        ax.axis("off")

plt.suptitle("Top PageRank Samples", fontsize=12)
plt.tight_layout()
plt.savefig(f"plots/outputs/passerines_pagerank_top_{top_n}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


# Get lowest-ranked indices
bottom_indices = sorted(pagerank_scores, key=pagerank_scores.get)[:top_n]

# Load corresponding images
bottom_images = [load_image(filenames[i]) for i in bottom_indices]
bottom_scores = [pagerank_scores[i] for i in bottom_indices]
bottom_names = [filenames[i] for i in bottom_indices]

# Display in dynamic grid (reuse previous dynamic grid logic)
n_cols = math.ceil(math.sqrt(top_n))
n_rows = math.ceil(top_n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
axes = axes.flatten()

for i in range(n_rows * n_cols):
    ax = axes[i]
    if i < len(bottom_images) and bottom_images[i] is not None:
        ax.imshow(bottom_images[i])
        ax.set_title(f"{bottom_names[i][:20]}...\n{bottom_scores[i]:.4f}", fontsize=8)
    else:
        ax.axis("off")

plt.suptitle("Least Influential Samples (Lowest PageRank)", fontsize=12)
plt.tight_layout()
plt.savefig(f"plots/outputs/passerines_pagerank_bottom_{top_n}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

