import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def get_thumbnail(path, size=(100, 100)):
    """Load and resize an image to create a thumbnail"""
    try:
        img = Image.open(path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return np.asarray(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


# === Step 1: Load embeddings CSV ===
csv_path = "ViT_embeddings/embeddings_config_kornia02_20250328T1803.csv" 
image_dir = "/Volumes/shared/cooney_lab/Shared/RSE/Image_sets/Passerines-Segmented-Back-RGB-PNG" 

# shape color embeddings MAE
# csv_path = "/Volumes/shared/cooney_lab/Shared/Eleftherios-Ioannou/mae/embeddings/mae_embeddings_20250407_104621_shape_color.csv" # "ViT_embeddings/embeddings_config_kornia02_20250328T1803.csv"  # <-- Update this path
# image_dir = "/Volumes/shared/cooney_lab/Shared/data/synth_data/shape_colour/images" 

# # synth data embeddings SimCLR
# csv_path = "/Volumes/shared/cooney_lab/Shared/Eleftherios-Ioannou/SimCLR/synthetic_data_embeddings/simclr/config_kornia02_20250404T1614/embeddings_config_kornia02_20250404T1614.csv"


csv_path = "/Volumes/shared/cooney_lab/Shared/Eleftherios-Ioannou/SimCLR/synthetic_data_embeddings/simclr/3808684_config_kornia02_20250411T1503/embeddings_config_kornia02_20250411T1503.csv"
image_dir = "/Volumes/shared/cooney_lab/Shared/data/synth_data/shape_colour_transparent" 

df = pd.read_csv(csv_path)

# === Select a random subset ===
subset_size = 5000
df_subset = df.sample(n=subset_size, random_state=42)  # ← random_state makes it reproducible


filenames = df_subset["filename"].values
X = df_subset.drop(columns=["filename"]).values.astype(np.float32)

# === Step 2: Diffusion map (Spectral Embedding) ===
gamma = 2  # (e.g. 0.01, 0.1, 10)
affinity = pairwise_kernels(X, metric="rbf", gamma=gamma)
embedding_2d = SpectralEmbedding(n_components=2, affinity='precomputed').fit_transform(affinity)

print("Embedding shape:", embedding_2d.shape)
print("Embedding sample:", embedding_2d[:5])
print("Min/max:", np.min(embedding_2d), np.max(embedding_2d))

# === Step 3: Plot with image thumbnails ===
def imscatter(x, y, thumbnails, ax=None, zoom=0.3, pad=0.0):
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

fig, ax = plt.subplots(figsize=(10, 8))

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

ax.set_title("Diffusion Map with Image Thumbnails", fontsize=18)
# ax.set_xticks([])
# ax.set_yticks([])
plt.xlabel("Diffusion Component 1")
plt.ylabel("Diffusion Component 2")
plt.grid(True)
# make the background black
ax.set_facecolor('black')
plt.tight_layout()
plt.savefig(f"ViT_embeddings/diffusion_maps/synthDataTransparent_diffusion_map_{gamma}_{subset_size}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
