import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import kornia
import kornia.augmentation as K
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys

# Add the lepidoptera directory to sys.path to import read_iml_hyp
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lepidoptera'))
from read_iml_hyp import read_iml_hyp, read_bil_file


class HyperspectralBilDataset(Dataset):
    """Dataset for loading hyperspectral BIL images with hundreds of spectral bands.
    Optimized for runtime loading with minimal preprocessing overhead.
    """
    
    def __init__(self, input_dir, transform=None, rgb_only=False, normalize=True, sample_bands=None):
        """
        Initialize the hyperspectral dataset.
        
        Args:
            input_dir (str): Directory containing the hyperspectral .bil files
            transform (callable, optional): Transform to apply to the data
            rgb_only (bool): If True, only return RGB-like bands (reduced dimensionality)
            normalize (bool): Whether to normalize the data
            sample_bands (list, optional): List of specific band indices to sample
                (useful for reducing dimensionality)
        """
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform
        self.rgb_only = rgb_only
        self.normalize = normalize
        self.sample_bands = sample_bands
        
        # Use glob for efficient file discovery (no nested loops)
        bil_files = glob.glob(os.path.join(input_dir, "**/*.bil"), recursive=True)
        
        # Filter to only include files with matching .bil.hdr (using set operations for efficiency)
        candidate_paths = []
        for bil_path in bil_files:
            base_path = bil_path[:-4]  # Remove .bil extension
            if os.path.exists(f"{base_path}.bil.hdr"):
                candidate_paths.append(base_path)
        
        # Sort the file paths for consistent order
        candidate_paths = sorted(candidate_paths)
        
        # Pre-filter files to only keep those that can be successfully loaded
        self.file_paths = []
        skipped_files = 0
        
        print(f"Pre-filtering {len(candidate_paths)} candidate files to remove problematic ones...")
        for path in candidate_paths:
            try:
                # Attempt to load the file to verify it works
                directory = os.path.dirname(path)
                filename = os.path.basename(path)
                # Quick validation - just try to load but don't store result
                read_iml_hyp(directory, filename)
                # If successful, add to valid files
                self.file_paths.append(path)
            except Exception as e:
                skipped_files += 1
                print(f"Skipping file {path}: {e}")
        
        print(f"Kept {len(self.file_paths)} valid files, skipped {skipped_files} problematic files")
        
        # Cache for wavelength data (will be loaded from first file)
        self.wavelengths = None
        
        # Cache common configurations to avoid repeated checks
        self._has_rgb_only = self.rgb_only
        self._has_sample_bands = self.sample_bands is not None
        self._normalize_data = self.normalize
        
        if self._has_rgb_only:
            print("Using RGB-like channels only (reducing dimensionality)")
        if self._has_sample_bands:
            print(f"Sampling {len(self.sample_bands)} specific bands")
        
        # Pre-calculate RGB indices if needed (to avoid repeated calculations in __getitem__)
        self.rgb_indices = None
        if self._has_rgb_only and len(self.file_paths) > 0:
            # Load the first file to get wavelengths for RGB indices calculation
            self._init_rgb_indices()
    
    def _init_rgb_indices(self):
        """Pre-calculate the RGB indices to avoid repeated calculations"""
        if len(self.file_paths) == 0:
            return
            
        # Extract directory and filename from the first path
        directory = os.path.dirname(self.file_paths[0])
        filename = os.path.basename(self.file_paths[0])
        
        try:
            # Only read wavelength data
            _, wavelengths, _, _ = read_iml_hyp(directory, filename)
            self.wavelengths = wavelengths
            
            # Find wavelengths closest to RGB (450nm, 550nm, 650nm)
            self.rgb_indices = []
            for target in [450, 550, 650]:
                idx = np.argmin(np.abs(wavelengths - target))
                self.rgb_indices.append(idx)
        except Exception as e:
            print(f"Warning: Could not precompute RGB indices: {e}")
            # Default to first 3 bands if wavelength data is unavailable
            self.rgb_indices = [0, 1, 2]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        
        try:
            # Extract directory and filename from the path
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            
            # Use read_iml_hyp to load the hyperspectral data
            # Note that in real processing we need the actual image data, not just wavelengths
            # The function may also return adjusted dimensions if file size doesn't match header
            image, wavelengths, scan, _ = read_iml_hyp(directory, filename)
            
            # Store wavelengths if not already cached
            if self.wavelengths is None:
                self.wavelengths = wavelengths
                # Initialize RGB indices if using rgb_only mode
                if self._has_rgb_only and self.rgb_indices is None:
                    # Find wavelengths closest to RGB (450nm, 550nm, 650nm)
                    self.rgb_indices = []
                    for target in [450, 550, 650]:
                        idx = np.argmin(np.abs(wavelengths - target))
                        self.rgb_indices.append(idx)
            
            # Apply early downsampling to reduce memory usage before any processing
            # This significantly reduces memory footprint for large hyperspectral images
            if image.shape[0] > 800 or image.shape[1] > 800:  # Only downsample large images
                # Calculate downsampling factor to get dimensions around 600-800 pixels
                h_factor = max(1, image.shape[0] // 600)
                w_factor = max(1, image.shape[1] // 600)
                factor = max(h_factor, w_factor)
                
                if factor > 1:
                    # Simple spatial downsampling (stride-based)
                    image = image[::factor, ::factor, :]
                    # print(f"Downsampled image from {image.shape[0]*factor}x{image.shape[1]*factor} to {image.shape[0]}x{image.shape[1]} (factor: {factor})")
            
            # Handle different band selection modes - using cached flags for efficiency
            if self._has_rgb_only:
                # Use pre-calculated RGB indices
                if self.rgb_indices is None:
                    self._init_rgb_indices()
                
                # Extract only RGB-like bands with vectorized operations
                selected_bands = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
                for i, idx in enumerate(self.rgb_indices):
                    if idx < image.shape[2]:
                        selected_bands[:, :, i] = image[:, :, idx].astype(np.float32)
            
            elif self._has_sample_bands:
                # Sample specific bands with vectorized operations where possible
                valid_bands = [b for b in self.sample_bands if b < image.shape[2]]
                band_count = len(valid_bands)
                
                selected_bands = np.zeros((image.shape[0], image.shape[1], band_count), dtype=np.float32)
                for i, band_idx in enumerate(valid_bands):
                    selected_bands[:, :, i] = image[:, :, band_idx].astype(np.float32)
            
            else:
                # Use all bands but switch to float32 early to reduce memory
                selected_bands = image.astype(np.float32)
                
                # Free the original uint16 data to save memory
                del image
            
            # Normalize the data if requested - used cached flag for efficiency
            if self._normalize_data:
                # Vectorized operations for normalization
                normalized_data = np.zeros_like(selected_bands, dtype=np.float32)
                for i in range(selected_bands.shape[2]):
                    band = selected_bands[:, :, i]
                    band_min = np.min(band)
                    band_max = np.max(band)
                    
                    # Avoid division by zero
                    if band_min == band_max:
                        normalized_data[:, :, i] = 0.0
                    else:
                        # Normalize to [0, 1] in one step
                        normalized_data[:, :, i] = (band - band_min) / (band_max - band_min)
                
                # Clip values to ensure they're in [0, 1]
                normalized_data = np.clip(normalized_data, 0.0, 1.0)
                
                # Convert to PyTorch tensor with channels first (C, H, W)
                tensor_data = torch.from_numpy(normalized_data.transpose(2, 0, 1)).float()
                
                # Free numpy array to save memory
                del normalized_data
            else:
                # Convert to PyTorch tensor with channels first (C, H, W)
                tensor_data = torch.from_numpy(selected_bands.transpose(2, 0, 1)).float()
                
                # Free numpy array to save memory
                del selected_bands
            
            # Handle NaN or Inf values to prevent downstream errors
            if torch.isnan(tensor_data).any() or torch.isinf(tensor_data).any():
                tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply transformations if provided
            # The transform should operate on the tensor_data only.
            # If self.transform is designed for SimCLR training, it might return a tuple of augmented views.
            # For prediction/embedding generation, it usually returns a single transformed tensor.
            if self.transform:
                transformed_data = self.transform(tensor_data) # Pass only the tensor to the transform
            else:
                transformed_data = tensor_data # No transform
            
            # The dataloader in simclr_birdcolour_kornia_hyperspectral.py expects (image_or_image_pair, filename)
            return transformed_data, filename

        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}")
            # Depending on how you want to handle errors, you might re-raise,
            # or return a placeholder that your collate_fn can filter out.
            # For now, re-raising to make the error explicit.
            raise RuntimeError(f"Failed to process {file_path}") from e


class HyperspectralGrayscale(torch.nn.Module):
    """Custom grayscale transform that works with hyperspectral images.
    
    This applies grayscale conversion by averaging across bands or by spectral groups.
    """
    
    def __init__(self, p=0.5, band_groups=None):
        """
        Args:
            p (float): Probability of applying the transform
            band_groups (list, optional): List of band index ranges to group together
                for grayscale conversion (e.g., [[0, 100], [100, 300], [300, 408]])
        """
        super().__init__()
        self.p = p
        self.band_groups = band_groups
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
                
        Returns:
            torch.Tensor: Grayscaled tensor with same shape
        """
        if torch.rand(1) > self.p:
            return x
        
        batch_size, channels, height, width = x.shape
        
        # If band groups are specified, apply grayscale by groups
        if self.band_groups is not None:
            output = torch.zeros_like(x)
            
            for start, end in self.band_groups:
                if end > channels:
                    end = channels
                
                if start >= channels:
                    continue
                
                # Calculate mean for this group of bands
                group_mean = torch.mean(x[:, start:end], dim=1, keepdim=True)
                
                # Repeat the mean across all channels in this group
                group_size = end - start
                group_expanded = group_mean.repeat(1, group_size, 1, 1)
                
                # Copy to output
                output[:, start:end] = group_expanded
            
            return output
        else:
            # Simple approach: average across all bands
            mean = torch.mean(x, dim=1, keepdim=True)
            return mean.repeat(1, channels, 1, 1)


class KorniaTransform:
    """Custom transform class that applies Kornia transforms."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        # Add batch dimension for Kornia
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        
        # Store original channel count to ensure it's preserved
        original_channels = img.shape[0] if len(img.shape) == 3 else img.shape[1]
        
        with torch.no_grad():
            if len(img.shape) == 3:
                # Add batch dimension
                img = img.unsqueeze(0)
                transformed = self.transform(img)
                
                # Check that channel count is preserved
                if transformed.shape[1] != original_channels:
                    print(f"WARNING: Channel count changed during transform: {original_channels} -> {transformed.shape[1]}")
                    # This shouldn't happen with a simple resize, but just in case
                
                return transformed.squeeze(0)
            else:
                transformed = self.transform(img)
                
                # Check that channel count is preserved
                if transformed.shape[1] != original_channels:
                    print(f"WARNING: Channel count changed during transform: {original_channels} -> {transformed.shape[1]}")
                
                return transformed


class KorniaSimCLRTransform:
    """Applies Kornia transforms to create two views of each image for contrastive learning."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, input_data):
        # input_data is expected to be a single tensor from HyperspectralBilDataset.__getitem__
        images = input_data
        
        # Kornia augmentations require a batch dimension. Check if we need to add one.
        needs_unsqueeze = len(images.shape) == 3
        if needs_unsqueeze:
            images = images.unsqueeze(0) # Add batch dimension: [C,H,W] -> [1,C,H,W]
        
        with torch.no_grad():
            # Create two augmented views
            # self.transform is the torch.nn.Sequential of Kornia augmentations
            view1 = self.transform(images)
            view2 = self.transform(images)
            
            # Remove the batch dimension if we added it
            if needs_unsqueeze:
                view1 = view1.squeeze(0)
                view2 = view2.squeeze(0)
            
            # Return only the tuple of view tensors
            return (view1, view2)


class HyperspectralDataModule(pl.LightningDataModule):
    """Lightning DataModule for hyperspectral data."""
    
    def __init__(self, config: dict, rgb_only: bool = False, sample_bands: list = None, augmentation_strategy: str = "crop"):
        super().__init__()
        self.config = config
        self.rgb_only = rgb_only
        self.sample_bands = sample_bands
        self.augmentation_strategy = augmentation_strategy
        
        # Get data configuration with fallbacks
        data_config = config.get("data", {})
        self.data_dir = data_config.get("data_dir")
        if not self.data_dir:
            raise ValueError("data_dir must be specified in the config file under the 'data' section")
        
        # Get dataloader configuration with fallbacks
        dataloader_config = config.get("dataloader", {})
        self.batch_size = dataloader_config.get("batch_size", 32)
        self.num_workers = dataloader_config.get("num_workers", 4)
        self.pin_memory = dataloader_config.get("pin_memory", True)
        self.prefetch_factor = dataloader_config.get("prefetch_factor", 2)
        
        self.input_size = config["augmentations"]["input_size"]
        self.transform_config = config["augmentations"]["transforms"]
        
        # Get transforms
        self.train_transform = self._get_train_transforms()
        
        # Initialize datasets to None
        self.train_dataset = None
        self.predict_dataset = None
        self.predict_transform = None
    
    def _get_train_transforms(self):
        """Create training augmentation pipeline using Kornia for hyperspectral data."""
        # Get transform configuration with fallbacks
        cfg = self.transform_config
        transforms = []
        

        
        
        # Add horizontal flip if enabled
        if cfg.get("horizontal_flip", {}).get("enabled", True):
            transforms.append(
                K.RandomHorizontalFlip(
                    p=cfg["horizontal_flip"]["p"]
                )
            )
        
        # Add vertical flip if enabled
        if cfg.get("vertical_flip", {}).get("enabled", True):
            transforms.append(
                K.RandomVerticalFlip(
                    p=cfg["vertical_flip"]["p"]
                )
            )
        
        # Add rotation if enabled
        if cfg.get("rotate", {}).get("enabled", True):
            transforms.append(
                K.RandomRotation(
                    degrees=cfg["rotate"]["limit"],
                    p=cfg["rotate"]["p"]
                )
            )
        
        # Add custom hyperspectral grayscale if enabled
        if cfg.get("grayscale", {}).get("enabled", True):
            if self.rgb_only:
                transforms.append(
                    K.RandomGrayscale(p=cfg["grayscale"]["p"])
                )
            else:
                # Define band groups for wavelength ranges (UV, visible, NIR)
                # These are approximate and can be adjusted based on specific wavelength ranges
                band_groups = [[0, 100], [100, 300], [300, 408]]
                transforms.append(
                    HyperspectralGrayscale(p=cfg["grayscale"]["p"], band_groups=band_groups)
                )
        
        # Add Gaussian blur if enabled
        if cfg.get("gaussian_blur", {}).get("enabled", True):
            transforms.append(
                K.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=cfg["gaussian_blur"]["sigma_limit"],
                    p=cfg["gaussian_blur"]["p"]
                )
            )
        
        # Add Gaussian noise if enabled
        if cfg.get("gaussian_noise", {}).get("enabled", True):
            transforms.append(
                K.RandomGaussianNoise(
                    mean=cfg["gaussian_noise"]["mean_range"][0],
                    std=cfg["gaussian_noise"]["std_range"][1],
                    p=cfg["gaussian_noise"]["p"]
                )
            )
        
        # Optional advanced transforms
        if "thin_plate_spline" in cfg and cfg["thin_plate_spline"]["enabled"]:
            transforms.append(
                K.RandomThinPlateSpline(
                    scale=cfg["thin_plate_spline"]["scale"],
                    p=cfg["thin_plate_spline"]["p"]
                )
            )
        
        if "random_perspective" in cfg and cfg["random_perspective"]["enabled"]:
            transforms.append(
                K.RandomPerspective(
                    distortion_scale=cfg["random_perspective"]["distortion_scale"],
                    p=cfg["random_perspective"]["p"]
                )
            )
        

        if self.augmentation_strategy == "crop":
            # Add random resize crop if enabled
            if cfg.get("random_resize_crop", {}).get("enabled", True):
                transforms.append(
                    K.RandomResizedCrop(
                        size=tuple(cfg["random_resize_crop"]["size"]),
                        scale=tuple(cfg["random_resize_crop"]["scale"]),
                        p=cfg["random_resize_crop"]["p"]
                    )
                )
            else:
                # Always resize if random resize crop is disabled
                transforms.append(
                    K.Resize(size=(self.input_size, self.input_size))
                )
        else:
            print("Using fixed size resize and padding")
            transforms.append(
                K.Resize(size=self.input_size, side='long')
            )
            # Pad the shorter side to make the image square
            transforms.append(
                K.PadTo(size=(self.input_size, self.input_size))
            )
        # Create a sequential transform
        return KorniaSimCLRTransform(torch.nn.Sequential(*transforms))
        # return KA.AugmentationSequential(*transforms, data_keys=["input"])
    
    def setup(self, stage=None):
        """Set up the datasets for training, validation, and testing."""
        if stage == 'fit' or stage is None:
            self.train_dataset = HyperspectralBilDataset(
                input_dir=self.data_dir,
                transform=self.train_transform,
                rgb_only=self.rgb_only,
                normalize=True,
                sample_bands=self.sample_bands
            )
        
        if stage == 'predict' or stage is None:
            # For prediction, we don't need SimCLR pairs
            predict_transforms = []
            # Always resize for prediction
            predict_transforms.append(
                K.Resize(size=(self.input_size, self.input_size))
            )
            
            self.predict_transform = K.AugmentationSequential(*predict_transforms, data_keys=["input"])
            
            self.predict_dataset = HyperspectralBilDataset(
                input_dir=self.data_dir,
                transform=KorniaTransform(self.predict_transform),
                rgb_only=self.rgb_only,
                normalize=True,
                sample_bands=self.sample_bands
            )
    
    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(self.num_workers, 2),  # Reduce number of workers to prevent OOM
            pin_memory=False,  # Disable pin_memory to reduce memory usage
            prefetch_factor=1,  # Reduce prefetch factor to save memory
            drop_last=True,
            collate_fn=self._collate_fn,
            persistent_workers=False  # Avoid keeping workers alive between epochs
        )
    
    def predict_dataloader(self):
        """Return the prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=max(1, self.batch_size // 2),  # Reduce batch size for prediction
            shuffle=False,
            num_workers=1,  # Use minimal workers for prediction to save memory
            pin_memory=False,
            prefetch_factor=1,
            collate_fn=self._predict_collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for handling SimCLR pairs."""
        # Debug: Print the shape of each batch item
        batch_valid = []
        for i, item in enumerate(batch):
            if isinstance(item, tuple) and len(item) == 2:
                # Check if this is a ((view1, view2), filename) structure
                views_tuple, filename = item
                if isinstance(views_tuple, tuple) and len(views_tuple) == 2:
                    batch_valid.append((views_tuple, filename))
                else:
                    print(f"WARNING: Skipping batch item {i}, expected a pair of tensors but got {type(views_tuple)}")
            else:
                print(f"WARNING: Skipping batch item {i} with unexpected format: {type(item)}, len={len(item) if isinstance(item, tuple) else 'N/A'}")
                continue

        if not batch_valid:
            raise ValueError("No valid items found in batch - all items had invalid formats")
            
        # Now use only the valid items
        view1_batch = []
        view2_batch = []
        filename_batch = []
        
        for (view1, view2), filename in batch_valid:
            view1_batch.append(view1)
            view2_batch.append(view2)
            filename_batch.append(filename)
        
        # Stack the batches
        view1_batch = torch.stack(view1_batch)
        view2_batch = torch.stack(view2_batch)
        
        # Return in format expected by SimCLR training_step: ((view1, view2), None)
        # The second item is a placeholder for labels (not used in contrastive learning)
        return (view1_batch, view2_batch), None
    
    def _predict_collate_fn(self, batch):
        """Custom collate function for prediction that ensures consistent batch format."""
        # Extract data and filenames from batch (no paired views in predict mode)
        data_batch = []
        filename_batch = []
        
        for item in batch:
            # Assuming each item is (data, filename) for prediction
            if isinstance(item, tuple) and len(item) == 2:
                data, filename = item
                data_batch.append(data)
                filename_batch.append(filename)
            else:
                print(f"WARNING: Skipping batch item with unexpected format: {type(item)}")
                continue
                
        if not data_batch:
            raise ValueError("No valid items found in prediction batch")
            
        # Stack the tensor data
        data_batch = torch.stack(data_batch)
        
        # Return in format expected by generate_embeddings: (img_pair, batch_filenames)
        return data_batch, filename_batch


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "data": {
            "data_dir": "/shared/cooney_lab/Shared/data/hyperspectral/lepidoptera"
        },
        "dataloader": {
            "batch_size": 16,
            "num_workers": 4
        },
        "augmentations": {
            "input_size": 224,
            "transforms": {
                "random_resize_crop": {
                    "enabled": True,
                    "size": [224, 224],
                    "scale": [0.8, 1.0],
                    "p": 1.0
                },
                "horizontal_flip": {
                    "enabled": True,
                    "p": 0.5
                },
                "vertical_flip": {
                    "enabled": True,
                    "p": 0.5
                },
                "rotate": {
                    "enabled": True,
                    "limit": 45,
                    "p": 0.5
                },
                "grayscale": {
                    "enabled": True,
                    "p": 0.2
                },
                "gaussian_blur": {
                    "enabled": True,
                    "sigma_limit": [0.1, 2.0],
                    "p": 0.5
                },
                "gaussian_noise": {
                    "enabled": True,
                    "mean_range": [0, 0],
                    "std_range": [0.01, 0.05],
                    "p": 0.5
                }
            }
        }
    }
    
    # Create a data module with options to use RGB-only or sample specific bands
    data_module = HyperspectralDataModule(
        config,
        rgb_only=False,
        sample_bands=None  # Use all bands by default
    )
    
    # Set up the data module
    data_module.setup()
    
    # Get a batch from the dataloader
    train_loader = data_module.predict_dataloader()
    for batch in train_loader:
        view1, view2 = batch
        print(f"View 1 shape: {view1.shape}, View 2 shape: {view2.shape}")
        break
