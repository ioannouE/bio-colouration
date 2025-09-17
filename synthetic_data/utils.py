import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import cv2
import os
import numpy as np
import glob

import pandas as pd

def load_mask_paths(mask_source, column=None, extensions=("*.png", "*.jpg", "*.tif", "*.tiff")):
    """
    Load mask paths from a folder or a DataFrame column.

    Args:
        mask_source: str (folder path) or pandas DataFrame
        column: str, if mask_source is a DataFrame, name of column with paths
        extensions: tuple of glob patterns to include
    Returns:
        List of absolute mask paths
    """
    mask_paths = []

    if isinstance(mask_source, str) and os.path.isdir(mask_source):
        for ext in extensions:
            mask_paths.extend(glob.glob(os.path.join(mask_source, ext)))
        mask_paths.sort()
    elif isinstance(mask_source, pd.DataFrame) and column:
        mask_paths = mask_source[column].dropna().tolist()
    else:
        raise ValueError("mask_source must be a folder path or DataFrame + column name.")
    
    return mask_paths

def read_binary_mask(path, threshold=0, keep_top_n=None):
    """
    Read a binary mask from a file, binarize it, and optionally keep only the top N largest components.

    Args:
        path: Path to the image file
        threshold: Threshold for binarization
        keep_top_n: int or None. If set, keeps only the N largest connected components.

    Returns:
        Binary mask as uint8 (0 or 1)
    """
    mask = cv2.imread(path, 0)
    if mask is None:
        raise IOError(f"Failed to read mask: {path}")
    
    # Threshold to binary
    binary = (mask > threshold).astype(np.uint8)

    if keep_top_n is not None:
        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return binary  # No foreground

        # Sort by area (skip background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        top_indices = np.argsort(areas)[::-1][:keep_top_n] + 1  # +1 to skip background
        filtered = np.isin(labels, top_indices).astype(np.uint8)
        return filtered

    return binary


# apply the mask and flatten to 1D
def apply_mask(image, mask):
    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask.astype(bool)
        elif mask.shape[-1] == 1:
            mask = mask.squeeze(-1).astype(bool)
        image = image[mask]
    else:
        image = image.reshape(-1, image.shape[-1])
    return image


def apply_mask_2d(image, mask=None):
    """
    Apply a mask to an image while maintaining the 2D structure.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        mask: Optional binary mask (H, W) or (H, W, 1)
    
    Returns:
        masked_image: Image with mask applied, maintaining original dimensions
    """
    if mask is None:
        return image.copy()
    
    # Ensure mask is boolean 2D
    if len(mask.shape) == 2:
        mask = mask.astype(bool)
    elif mask.shape[-1] == 1:
        mask = mask.squeeze(-1).astype(bool)
    
    # For RGB images
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Create an RGBA image (with transparency)
        h, w, _ = image.shape
        masked_image = np.zeros((h, w, 4), dtype=image.dtype)
        # Copy RGB channels
        masked_image[:, :, :3] = image
        # Set alpha channel based on mask (255 for visible, 0 for transparent)
        masked_image[:, :, 3] = np.where(mask, 255, 0)
    # For grayscale images
    else:
        # Create a copy of the image
        masked_image = np.zeros_like(image)
        # Apply the mask
        masked_image[mask] = image[mask]
    
    # also crop only where there are pixels both in width and height
    masked_image = masked_image[mask.any(axis=1)]
    # and crop on the sides
    masked_image = masked_image[:, mask.any(axis=0)]
   
    # return the masked image with RGB channels
    return masked_image



###### Resize the image
def resize_with_aspect(image, mask=None, max_size=512, interpolation=cv2.INTER_AREA):
    """
    Resize image (and mask if given) so that the longest side is <= max_size.
    Maintains aspect ratio.
    """
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    
    if scale >= 1.0:
        # No resizing needed
        return image, mask
    
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if mask is not None:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask
    else:
        return resized_image, None


############## For visualisation ############
# def resize_keep_aspect(img, longest_side):
#     h, w = img.shape[:2]
#     if h >= w:
#         new_h = longest_side
#         new_w = int(w * (longest_side / h))
#     else:
#         new_w = longest_side
#         new_h = int(h * (longest_side / w))
#     return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def create_masked_rgba_thumbnail(img, mask, thumb_size=(40, 40), alpha_fg=0.7, alpha_bg=0.0):
    """
    Create an RGBA thumbnail where pixels outside the mask have lower alpha.
    """
    # Crop to bounding box
    ys, xs = np.where(mask)
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    img_cropped = img[y_min:y_max+1, x_min:x_max+1]
    mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]

    # Resize
    img_resized, _ = resize_with_aspect(img_cropped, max_size=thumb_size[0])
    mask_resized, _ = resize_with_aspect(mask_cropped, max_size=thumb_size[0])

    # Normalize mask to binary
    mask_binary = (mask_resized > 0).astype(np.uint8)

    # Create RGBA image
    rgba = np.ones((*img_resized.shape[:2], 4), dtype=np.uint8) * 255  # white background

    rgba[:, :, :3] = img_resized
    rgba[:, :, 3] = np.where(mask_binary, int(alpha_fg * 255), int(alpha_bg * 255))

    return rgba


def normalize_df_columns(df, columns, inplace=True):
    """
    Normalize specified columns of a DataFrame to range [0, 1].

    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        inplace: if True, modify df in-place. If False, return a new DataFrame.

    Returns:
        normalized DataFrame (if inplace=False)
    """
    target_df = df if inplace else df.copy()

    for col in columns:
        min_val = target_df[col].min()
        max_val = target_df[col].max()
        range_val = max_val - min_val + 1e-8  # avoid div-by-zero
        target_df[col] = (target_df[col] - min_val) / range_val

    return target_df if not inplace else None


def scatter_with_thumbnails(
    df,
    x_col,
    y_col,
    image_dir,
    image_col='image_name',
    mask_dir=None,
    mask_col='mask_name',
    thumb_size=(40, 40),
    ax=None,  # <- add ax here
    figsize = (10, 8),
    dpi=150,
    output_path=None,
    normalize=False,
    skewed_fix = False,
    title = None,
    x_label=None,
    y_label=None,
    plot_show = True
):
    """
    Scatter plot of two numeric columns with image thumbnails,
    optionally cropped by masks before resizing.

    Args:
        df: pandas DataFrame with data
        x_col, y_col: column names to use as scatter coordinates
        image_dir: folder containing RGB images
        image_col: column with image file names
        mask_dir: optional folder containing masks (grayscale PNGs)
        mask_col: column with mask file names (must match mask_dir)
        thumb_size: (w, h) for thumbnail size in pixels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize , dpi = dpi)

    # === Font size scaling ===
    base_font = 10
    scale_factor = (figsize[0] * figsize[1]) / (10 * 8)  # scaled relative to (10,8)
    font_size = base_font * scale_factor

    df = df.copy()

    if normalize:
        normalize_df_columns(df, [x_col, y_col])


    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
     
    ax.scatter(x, y, alpha=0.0)

    for i, row in df.iterrows():
        img_path = os.path.join(image_dir, row[image_col])
        if not os.path.isfile(img_path):
            continue

        # Load image
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to read image: {img_path} – {e}")
            continue

        # Load and apply mask if provided
        if mask_dir is not None and mask_col in row:
            mask_path = os.path.join(mask_dir, row[mask_col])
            if os.path.isfile(mask_path):
                try:
                    mask = cv2.imread(mask_path, 0)
                    mask = (mask > 0).astype(np.uint8)

                    # Apply bounding box crop
                    ys, xs = np.where(mask)
                    if len(xs) > 0 and len(ys) > 0:
                        x_min, x_max = np.min(xs), np.max(xs)
                        y_min, y_max = np.min(ys), np.max(ys)
                        img = img[y_min:y_max+1, x_min:x_max+1]
                        mask = mask[y_min:y_max+1, x_min:x_max+1]

                        # Optional: apply mask to crop transparency (if needed)
                        # img = cv2.bitwise_and(img, img, mask=mask)
                        # Create white background
                        white_bg = np.ones_like(img, dtype=np.uint8) * 255

                        # Copy foreground where mask > 0
                        for c in range(3):
                            white_bg[:, :, c] = np.where(mask > 0, img[:, :, c], 255)

                        img = white_bg
                        
                except Exception as e:
                    print(f"Failed to process mask for {img_path}: {e}")

        # Resize thumbnail
        try:
            # thumb = cv2.resize(img, thumb_size)
            if mask_dir is not None:
                rgba_thumb = create_masked_rgba_thumbnail(img, mask, thumb_size, alpha_fg=0.8, alpha_bg=0.0)
                
                imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(rgba_thumb, zoom=1),  # ← adjust alpha here
                (row[x_col], row[y_col]),
                frameon=False,
                pad=0.1
            )       
            else:
                thumb = cv2.resize(img, thumb_size)
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(thumb, zoom=1),
                    (row[x_col], row[y_col]),
                    frameon=False,
                    pad=0.1
                )
            
     


            ax.add_artist(imagebox)
        except Exception as e:
            print(f"Thumbnail failed for {img_path}: {e}")

    if x_label is None:
        x_label = x_col 
    if y_label is None:
        y_label = y_col

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if title is None:
        title = f"Scatter: {x_col} vs {y_col}"
    if normalize:
        title += " (normalized)"
    ax.set_title(title)
    
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_title(title, fontsize=font_size * 1.2)
    ax.tick_params(axis='both', labelsize=font_size * 0.9)
    
    ax.grid(True)
    plt.tight_layout()
    
    if output_path is not None:
        try:
            plt.savefig(output_path, dpi=300)
            print(f"[INFO] Plot saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save plot to {output_path}: {e}")
    elif plot_show:
        plt.show()