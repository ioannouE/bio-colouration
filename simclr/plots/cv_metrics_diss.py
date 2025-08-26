import warnings
warnings.filterwarnings("ignore")
import os
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms, models
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
from tqdm import tqdm
from umap_plot import get_thumbnail, imscatter
# do parallel thread execution
from concurrent.futures import ThreadPoolExecutor
# Import for SSIM
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.manifold import MDS

# ---------- Helper Functions ----------

def apply_mask(image, mask):
    # Convert to numpy arrays for manipulation
    img_np = np.array(image)
    mask_np = np.array(mask)
    
    # Binarize the mask if it's not already binary
    mask_np = (mask_np > 0).astype(np.uint8)
    
    # Apply mask - set background (where mask is 0) to white or another color
    # Expand mask to 3 channels to match RGB image
    mask_np_3ch = np.stack([mask_np, mask_np, mask_np], axis=2)
    
    # Apply the mask: keep foreground pixels, set background to white
    background = np.ones_like(img_np) * 255  # White background
    masked_img = np.where(mask_np_3ch == 1, img_np, background)
    
    # Convert back to PIL Image
    return Image.fromarray(masked_img.astype(np.uint8))


def load_images_from_folder(folder, mask_dir=None, image_size=256, max_images=100):
    images = []
    image_paths = []
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1]
    ])
    
    file_list = [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    if max_images and max_images < len(file_list):
        file_list = file_list[:max_images]
    
    # Define a function to process a single image
    def process_image(filename):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert('RGB')
            if mask_dir:
                try:
                    mask_path = os.path.join(mask_dir, filename)
                    mask = Image.open(mask_path+'.png').convert('L')  # Convert to grayscale
                    
                    # Resize mask to match the image if needed
                    if mask.size != img.size:
                        mask = mask.resize(img.size, Image.Resampling.NEAREST)
                    # Apply mask
                    img = apply_mask(img, mask)
                except Exception as e:
                    print(f"Error applying mask to {filename}: {e}")
            img_tensor = transform(img)
            return img_tensor, path
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            return None, None
    
    # Use ThreadPoolExecutor to load and process images in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(process_image, file_list),
            total=len(file_list),
            desc="Loading images",
            colour='green'
        ))
    
    # Filter out None results and separate images and paths
    for img_tensor, path in results:
        if img_tensor is not None:
            images.append(img_tensor)
            image_paths.append(path)
    
    return images, image_paths


def compute_distance_matrix(images, metric='lpips', lpips_model=None, batch_size=100):
    """
    Compute a distance matrix using the specified metric.
    
    Args:
        images: List of image tensors
        metric: 'lpips' or 'ssim'
        lpips_model: LPIPS model (required if metric='lpips')
        batch_size: Batch size for processing
        
    Returns:
        Distance matrix as numpy array
    """
    n = len(images)
    D = np.zeros((n, n))
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device if using LPIPS
    if metric == 'lpips' and lpips_model is not None:
        lpips_model = lpips_model.to(device)
    
    if metric == 'style':
        vgg = models.vgg19(pretrained=True).features.to(device)
        vgg.eval()
    
    if metric == 'combine':
        lpips_model = lpips_model.to(device)
        vgg = models.vgg19(pretrained=True).features.to(device)
        vgg.eval()

    # Define a function to compute a single distance pair
    def compute_distance(pair):
        i, j = pair
        
        if metric == 'lpips':
            with torch.no_grad():
                # Move images to the same device as the model
                img_i = images[i].unsqueeze(0).to(device)
                img_j = images[j].unsqueeze(0).to(device)
            
                dist = lpips_model(img_i, img_j)
                # Explicitly free GPU memory
                torch.cuda.empty_cache()
                return (i, j, dist.item())
                
        
        elif metric == 'ssim':
            # Convert tensors to numpy arrays for SSIM
            img_i = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
            img_j = images[j].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize to [0, 1] range for SSIM if needed
            if img_i.min() < 0 or img_i.max() > 1:
                img_i = (img_i + 1) / 2.0  # Convert from [-1,1] to [0,1]
            if img_j.min() < 0 or img_j.max() > 1:
                img_j = (img_j + 1) / 2.0
                
            # Calculate SSIM (returns similarity, so convert to distance)
            # Specify data_range=1.0 since we normalized to [0,1]
            similarity = ssim(img_i, img_j, multichannel=True, channel_axis=2, data_range=1.0)
            distance = 1.0 - similarity  # Convert similarity to distance
            return (i, j, distance)
        elif metric == 'psnr':
            # Convert tensors to numpy arrays for PSNR
            img_i = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
            img_j = images[j].cpu().numpy().transpose(1, 2, 0)
            
            # Normalize to [0, 1] range for PSNR if needed
            if img_i.min() < 0 or img_i.max() > 1:
                img_i = (img_i + 1) / 2.0  # Convert from [-1,1] to [0,1]
            if img_j.min() < 0 or img_j.max() > 1:
                img_j = (img_j + 1) / 2.0
                
            # Calculate PSNR
            psnr_value = psnr(img_i, img_j)
            return (i, j, psnr_value)
        elif metric == 'style':
            # calculate style loss using gram matrix
            # extract features
            img_i = images[i].unsqueeze(0).to(device)
            img_j = images[j].unsqueeze(0).to(device)
            
            style_features_i = get_features(img_i, vgg)
            style_grams_i = {layer: gram_matrix(style_features_i[layer]) for layer in style_features_i}
            
            style_features_j = get_features(img_j, vgg)
            style_grams_j = {layer: gram_matrix(style_features_j[layer]) for layer in style_features_j}
            
            style_weights = {'conv1_1': 1.,
                                'conv2_1': 0.8,
                                'conv3_1': 0.5,
                                'conv4_1': 0.3,
                                'conv5_1': 0.1}
            
            style_loss = 0
            for layer in style_weights:
                # style_loss += style_weights[layer] * torch.mean((style_grams_i[layer] - style_grams_j[layer]) ** 2)
                style_loss += torch.mean((style_grams_i[layer] - style_grams_j[layer]) ** 2)
            
            return (i, j, style_loss.item())

        elif metric == 'combine':
            # extract features
            img_i = images[i].unsqueeze(0).to(device)
            img_j = images[j].unsqueeze(0).to(device)

            with torch.no_grad():
                lpips_loss = lpips_model(img_i, img_j)

            style_weights = {'conv1_1': 1.,
                                'conv2_1': 0.8,
                                'conv3_1': 0.5,
                                'conv4_1': 0.3,
                                'conv5_1': 0.1}
          
            
            style_features_i = get_features(img_i, vgg)
            style_grams_i = {layer: gram_matrix(style_features_i[layer]) for layer in style_features_i}
            
            style_features_j = get_features(img_j, vgg)
            style_grams_j = {layer: gram_matrix(style_features_j[layer]) for layer in style_features_j}
            
            style_loss = 0
            for layer in style_weights:
                # style_loss += style_weights[layer] * torch.mean((style_grams_i[layer] - style_grams_j[layer]) ** 2)
                style_loss += torch.mean((style_grams_i[layer] - style_grams_j[layer]) ** 2)
            
            return (i, j, (style_loss.item() + lpips_loss.item()) / 2)
            
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    # Generate indices more memory efficiently - don't create the full list at once
    total_pairs = n * (n - 1) // 2
    
    # Use ThreadPoolExecutor to compute distances in parallel
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        # Create a progress bar first
        pbar = tqdm(total=total_pairs, desc=f"Computing {metric.upper()} distances", colour='magenta')
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, n):
            # Create pairs for this batch
            batch_pairs = [(batch_start, j) for j in range(batch_start+1, n)]
            
            if not batch_pairs:
                continue
                
            # Process results as they complete
            for i, j, dist in executor.map(compute_distance, batch_pairs):
                D[i, j] = D[j, i] = dist
                pbar.update(1)
                
            # Force garbage collection after each batch
            import gc
            gc.collect()
            if metric == 'lpips':
                torch.cuda.empty_cache()
        
        # Close the progress bar
        pbar.close()
        
    return D


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    if layers is None:
        layers = {'0': 'conv1_1',
                 '5':  'conv2_1',
                 '10': 'conv3_1',
                 '19': 'conv4_1',
                 '21': 'conv4_2',
                 '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    gram = None
    b, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def plot_embedding_with_thumbnails(embedding, image_paths, mask_dir=None, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    
    # Add scatter points for each embedding position
    # ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.2, c='blue')
    
    # Set axis limits with some padding
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add image thumbnails
    for xy, path in zip(embedding, image_paths):
        try:
            if mask_dir: 
                img = apply_mask(get_thumbnail(path), Image.open(os.path.join(mask_dir, os.path.basename(path+'.png'))))
            img = get_thumbnail(path)
            imagebox = OffsetImage(img, zoom=0.5)
            ab = AnnotationBbox(imagebox, xy, frameon=False, box_alignment=(0.5, 0.5), pad=0.1)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error displaying image {path}: {e}")
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ---------- Main Program ----------

def main(args):
    # Load images
    print(f"Loading images from {args.images}...")
    images, image_paths = load_images_from_folder(args.images, args.masks, max_images=args.max_images)
    print(f"Loaded {len(images)} images.")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model based on metric
    metric = args.metric.lower()
    lpips_model = None
    
    if metric == 'lpips' or metric == 'combine':
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='vgg')
    
    # Compute distance matrix
    print(f"Computing {metric.upper()} distances...")
    distance_matrix = compute_distance_matrix(images, metric=metric, lpips_model=lpips_model)
    
    # Save the distance matrix along with the filenames as a CSV file
    print("Saving distance matrix with filenames...")
    # Extract just the filenames without the full path
    filenames = [os.path.basename(path) for path in image_paths]
    
    # Create a DataFrame with filenames as both row and column headers
    import pandas as pd
    df = pd.DataFrame(distance_matrix, index=filenames, columns=filenames)
    
    # Save to CSV
    output_file = f"{metric}_distance_matrix.csv"
    df.to_csv(output_file)
    print(f"Distance matrix saved to {output_file}")

    # Apply UMAP
    print("Running UMAP embedding...")
    reducer_umap = umap.UMAP(metric='precomputed', random_state=42)
    embedding_umap = reducer_umap.fit_transform(distance_matrix)
    
    # Plot UMAP
    print("Plotting UMAP result...")
    plot_embedding_with_thumbnails(embedding_umap, image_paths, title=f"UMAP - {metric.upper()} Embedding", save_path=f"umap_{metric}_plot.png")

    # Apply t-SNE
    print("Running t-SNE embedding...")
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
    embedding_tsne = tsne.fit_transform(distance_matrix)

    # Plot t-SNE
    print("Plotting t-SNE result...")
    plot_embedding_with_thumbnails(embedding_tsne, image_paths, title=f"t-SNE - {metric.upper()} Embedding", save_path=f"tsne_{metric}_plot.png")

    # Plot using MDS
    print("Running MDS embedding...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding_mds = mds.fit_transform(distance_matrix)

    # Plot MDS
    print("Plotting MDS result...")
    plot_embedding_with_thumbnails(embedding_mds, image_paths, title=f"MDS - {metric.upper()} Embedding", save_path=f"mds_{metric}_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptual embedding with UMAP and t-SNE visualization.")
    parser.add_argument('--images', type=str, required=True, help="Path to folder containing images.")
    parser.add_argument('--masks', type=str, required=False, help="Path to folder containing masks.")
    parser.add_argument('--max_images', type=int, required=False, default=1000, help="Maximum number of images to load.")
    parser.add_argument('--metric', type=str, required=False, default='lpips', choices=['lpips', 'ssim', 'psnr', 'style', 'combine'], help="Distance metric to use (lpips or ssim).")
    args = parser.parse_args()
    main(args)
