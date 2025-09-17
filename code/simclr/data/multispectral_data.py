import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, List, Tuple, Union
import kornia
import kornia.augmentation as K
import pytorch_lightning as pl
from PIL import Image
import cv2


def rgb_uv_to_tetrahedral(vR, vG, vB, uR, uB):
    """
    Convert RGB+UV light measurements to tetrahedral color space.
    
    Args:
        vR, vG, vB: Visible light components (Red, Green, Blue)
        uR, uB: UV light components (UV Red, UV Blue)
        
    Returns:
        numpy array with components [uv, sw, mw, lw, dbl] in shape [5, H, W]
    """
    # UV component
    uv = (-1.0072645856543222E-4 + 
          (vR * 2.656681655868963E-4) + 
          (vG * -5.068912001434062E-4) + 
          (vB * 8.807031837400417E-4) + 
          (uB * -8.51767628459907E-4) + 
          (uR * 0.010001038089545292) + 
          (vR * vG * -1.7442166408753685E-6) + 
          (vR * vB * -4.072619129161714E-6) + 
          (vR * uB * 3.5552224437902936E-5) + 
          (vR * uR * -5.4142649401885054E-5) + 
          (vG * vB * 2.7000549174292213E-6) + 
          (vG * uB * -8.844846862608557E-5) + 
          (vG * uR * 1.2402794993867318E-4) + 
          (vB * uB * 5.712851064047926E-5) + 
          (vB * uR * -6.157175168330311E-5) + 
          (uB * uR * -8.51608159488413E-6))
    
    # Short wavelength
    sw = (-3.229029565299134E-5 + 
          (vR * -1.7026643205230075E-4) + 
          (vG * -6.807783829936864E-4) + 
          (vB * 0.0108993521112189) + 
          (uB * -4.5212882213361216E-4) + 
          (uR * 3.9755680570539853E-4) + 
          (vR * vG * 1.132946264533436E-6) + 
          (vR * vB * -5.609632726275387E-7) + 
          (vR * uB * 3.1010255525254463E-6) + 
          (vR * uR * -4.513028334272119E-6) + 
          (vG * vB * 7.230150619585592E-7) + 
          (vG * uB * 3.753211813835518E-5) + 
          (vG * uR * -3.200234702080414E-5) + 
          (vB * uB * -4.5045649541170734E-5) + 
          (vB * uR * 4.229031861293349E-5) + 
          (uB * uR * -2.6632371305764216E-6))
    
    # Medium wavelength
    mw = (1.229975364943929E-4 + 
          (vR * -0.0015216879782443656) + 
          (vG * 0.016739432462496474) + 
          (vB * -0.00511190123344914) + 
          (uB * 4.236292426241492E-4) + 
          (uR * -4.2627071710053485E-4) + 
          (vR * vG * -2.5821513349848594E-8) + 
          (vR * vB * 1.9647622682695667E-6) + 
          (vR * uB * -2.240636669502069E-6) + 
          (vR * uR * -1.4422910213585519E-6) + 
          (vG * vB * -5.359352040828575E-6) + 
          (vG * uB * -7.647244014122641E-5) + 
          (vG * uR * 7.518474561971779E-5) + 
          (vB * uB * 8.085726066051999E-5) + 
          (vB * uR * -7.489876104635781E-5) + 
          (uB * uR * 1.3240918187677544E-6))
    
    # Long wavelength
    lw = (-3.2674106020576883E-4 + 
          (vR * 0.0086361781366186) + 
          (vG * 0.003940421038090461) + 
          (vB * -0.0030333184099389106) + 
          (uB * -5.279319408751551E-4) + 
          (uR * 5.668779542889099E-4) + 
          (vR * vG * -1.0396031692336333E-5) + 
          (vR * vB * 3.408196259708747E-6) + 
          (vR * uB * 3.325090506155861E-5) + 
          (vR * uR * -1.8568872302079177E-5) + 
          (vG * vB * 1.4233420422229957E-5) + 
          (vG * uB * 2.5609228198758987E-5) + 
          (vG * uR * -5.2131306767304E-5) + 
          (vB * uB * -4.057759093927875E-5) + 
          (vB * uR * 4.2991263255130344E-5) + 
          (uB * uR * 7.3468699809824446E-6))
    
    # Double cone
    dbl = (-1.0544979721066653E-4 + 
           (vR * 0.0014709063997941974) + 
           (vG * 0.010378782427678692) + 
           (vB * -0.0020526053269969012) + 
           (uB * -2.6958873663153905E-4) + 
           (uR * 3.7353984492219025E-4) + 
           (vR * vG * -2.980840024845194E-7) + 
           (vR * vB * -3.419399239171408E-6) + 
           (vR * uB * 1.6427954610944008E-5) + 
           (vR * uR * -1.1974755559207828E-5) + 
           (vG * vB * 5.8077800980947696E-6) + 
           (vG * uB * 3.627998939567357E-6) + 
           (vG * uR * -8.536616709636622E-6) + 
           (vB * uB * -1.5150428165844035E-5) + 
           (vB * uR * 1.2798214483425781E-5) + 
           (uB * uR * 1.9061233794070093E-6))
    
    return np.array([uv, sw, mw, lw, dbl])



class MultispectralTifDataset(torch.utils.data.Dataset):
    """Dataset for loading multispectral TIF images with 6 spectral bands and a mask layer.
    Optimized for runtime loading with minimal preprocessing overhead."""
    
    def __init__(self, input_dir, transform=None, rgb_only=False, uv_only=False, usml=False, channels=None):
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform
        self.rgb_only = rgb_only
        self.uv_only = uv_only
        self.usml = usml
        self.channels = channels
        
        # Use glob for efficient file discovery (no nested loops)
        self.filenames = sorted(glob.glob(os.path.join(input_dir, "**/*.tif*"), recursive=True))
        
        # Pre-open file handles for faster access
        self.file_handles = {}
        
        print(f"Found {len(self.filenames)} TIF files in {input_dir}")
        if self.rgb_only:
            print("Using RGB channels only (ignoring UV data)")
        if self.uv_only:
            print("Using UV channels only (ignoring RGB data)")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        """Load data at runtime with minimal overhead."""
        filename = self.filenames[index]
        
        # Open the file only when needed
        with rasterio.open(filename) as src:
            # Read data directly into memory with a single read operation
            # This avoids nested loops and multiple read operations
            image = src.read()
            
            # image = np.clip(image / 65535.0, 0, 1)

            # Extract spectral bands and mask in a vectorized operation
            if self.rgb_only:
                # Use only the first 3 channels (RGB) when rgb_only is True
                spectral_data = image[:3]
                # print("RGB shape: ", spectral_data.shape)
            elif self.uv_only:
                # Use only the last 3 channels (UV) when uv_only is True
                spectral_data = image[3:6]
            elif self.channels is not None:
                spectral_data = image[self.channels]
            else:
                # Use all 6 channels (RGB + UV) when rgb_only is False
                spectral_data = image[:6]  # First 6 bands (3 visible + 3 UV)
            mask = image[6:7] if image.shape[0] > 6 else None
            
            # Apply mask to spectral data
            if mask is not None:
                spectral_data = np.where(mask == 1, spectral_data, 0)

            # Apply USML
            if self.usml:
                vR, vG, vB = spectral_data[0], spectral_data[1], spectral_data[2]
                uR, uB = spectral_data[3], spectral_data[4]
                spectral_data = rgb_uv_to_tetrahedral(vR, vG, vB, uR, uB)
                # print("USML applied shape: ", spectral_data.shape)
                # spectral_data = np.expand_dims(spectral_data, axis=0)  # Add channel dimension

            # if np.isnan(spectral_data).any() or np.isinf(spectral_data).any():
            #     # print(f"Warning: File {filename} contains NaN or infinite values. Replacing with zeros.")
            #     spectral_data = np.nan_to_num(spectral_data, nan=0.0, posinf=0.0, neginf=0.0)

          
            # # # Proper normalization for each band separately
            normalized_data = np.zeros_like(spectral_data, dtype=np.float32)
            for i in range(spectral_data.shape[0]):
                band = spectral_data[i].astype(np.float32)
                
                # Get min and max for this band, with safeguards
                band_min = np.min(band)
                band_max = np.max(band)
                
                # Avoid division by zero
                if band_min == band_max:
                    normalized_data[i] = 0.0
                else:
                    # Normalize to [0, 1]
                    normalized_data[i] = (band - band_min) / (band_max - band_min)
            
            # Clip values to ensure they're in [0, 1]
            spectral_data = np.clip(spectral_data, 0.0, 1.0)
            
          

            # Convert to tensor (already in correct C,H,W format)
            spectral_tensor = torch.from_numpy(spectral_data).float()
            
            # Apply transforms if available
            if self.transform:
                transformed_tensor = self.transform(spectral_tensor)
                
                # print("Max transformed: ", torch.max(transformed_tensor[0]))
                # print("Min transformed: ", torch.min(transformed_tensor[0]))

                # # Check for NaN values after transform
                # if torch.isnan(spectral_tensor).any():
                #     print(f"Warning!!!!!: After transform, tensor from {filename} contains NaN values. Replacing with zeros.")
                #     transformed_tensor = torch.nan_to_num(spectral_tensor, nan=0.0)
            
            # Return the tensor, and the filename
            return transformed_tensor, os.path.basename(filename)
            
    def __del__(self):
        """Clean up any open file handles."""
        for handle in self.file_handles.values():
            if handle is not None:
                handle.close()





def collate_fn(batch):
    """Custom collate function to handle tuple of tensors"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Each item in batch should be (tensor_pair, filename)
    tensor_pairs, filenames = zip(*batch)
    
    # Each tensor_pair should be (aug1, aug2)
    aug1s, aug2s = [], []
    for pair in tensor_pairs:
        if isinstance(pair, tuple) and len(pair) == 2:
            aug1, aug2 = pair
            aug1s.append(aug1)
            aug2s.append(aug2)
        else:
            print(f"Warning: Unexpected tensor pair format: {type(pair)}")
            continue
    
    if not aug1s or not aug2s:
        raise ValueError("No valid augmented pairs found in batch")
    
    # Stack the tensors
    aug1s = torch.stack(aug1s)
    aug2s = torch.stack(aug2s)
    
    return (aug1s, aug2s), filenames



class KorniaTransform:
    """Custom transform class that applies Kornia transforms twice for SimCLR"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = img.unsqueeze(0)  # Add batch dimension for Kornia
        
        # Apply transforms twice to get two augmented versions
        try:
            result1 = self.transform(img).squeeze(0)
            result2 = self.transform(img).squeeze(0)
            
            # Check for NaN values
            if torch.isnan(result1).any():
                result1 = torch.nan_to_num(result1, nan=0.0)
            if torch.isnan(result2).any():
                result2 = torch.nan_to_num(result2, nan=0.0)
                
            return (result1, result2)
        except Exception as e:
            print(f"Error in transform: {e}")
            # Return the original image twice if transform fails
            return (img.squeeze(0), img.squeeze(0))


class MultispectralGrayscale(K.AugmentationBase2D):
    """Custom grayscale transform that works with multispectral (6-channel) images.
    
    This applies grayscale conversion separately to the RGB and UV channels.
    """
    
    def __init__(self, p: float = 0.5, same_on_batch: bool = False, keepdim: bool = False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        
    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any], transform=None) -> torch.Tensor:
    
        # For 6-channel images, apply grayscale separately to RGB and UV channels
        rgb_channels = input[:, :3]
        uv_channels = input[:, 3:6]
        
        # Apply grayscale to RGB channels
        rgb_gray = kornia.color.rgb_to_grayscale(rgb_channels)
        rgb_gray = rgb_gray.repeat(1, 3, 1, 1)  # Expand back to 3 channels
        
        # Apply grayscale to UV channels
        uv_gray = kornia.color.rgb_to_grayscale(uv_channels)
        uv_gray = uv_gray.repeat(1, 3, 1, 1)  # Expand back to 3 channels
        
        # Combine the results
        return torch.cat([rgb_gray, uv_gray], dim=1)
       

# Define a picklable transform class
class PredictTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        result = self.transform(x.unsqueeze(0)).squeeze(0)
        
        # If the transform returns a tuple, extract the first element
        if isinstance(result, tuple):
            result = result[0]
            
        # Check for NaN values
        if torch.isnan(result).any():
            result = torch.nan_to_num(result, nan=0.0)
            
        return result

class SimCLRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict,
        rgb_only: bool = False,
        uv_only: bool = False,
        usml: bool = False,
        channels=None,
    ):
        super().__init__()
        self.config = config
        self.rgb_only = rgb_only
        self.uv_only = uv_only
        self.usml = usml
        self.channels = channels
        self.augmentation_strategy = config.get("augmentations", {}).get(
            "strategy", "crop"
        )
        
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
        self.predict_transform = None  # Store the predict transform

    def _get_train_transforms(self):
        """Create training augmentation pipeline using Kornia for spectral data."""
        # Get transform configuration with fallbacks
        cfg = self.transform_config
        transforms = []
        
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

            # if self.augmentation_strategy == "crop":
            #     # Always resize if random resize crop is disabled
            #     transforms.append(
            #         K.Resize(size=(self.input_size, self.input_size))
            #     )
            # else:
            print("Random resize crop is disabled, using fixed size resize and padding")
            transforms.append(
                K.Resize(size=self.input_size, side='long')
            )
            # Pad the shorter side to make the image square
            transforms.append(
                K.PadTo(size=(self.input_size, self.input_size))
            )
        
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
        
        # Add color jitter if enabled
        if cfg.get("color_jitter", {}).get("enabled", True):
            transforms.append(
                K.ColorJitter(
                    brightness=cfg["color_jitter"]["brightness"],
                    contrast=cfg["color_jitter"]["contrast"],
                    saturation=cfg["color_jitter"]["saturation"],
                    hue=cfg["color_jitter"]["hue"],
                    p=cfg["color_jitter"]["p"]
                )
            )
        
       
        # Add grayscale if enabled
        if cfg.get("grayscale", {}).get("enabled", True):
            if self.rgb_only:
                transforms.append(
                    K.RandomGrayscale(p=cfg["grayscale"]["p"])
                )
            else:
                 transforms.append(
                    MultispectralGrayscale(p=cfg["grayscale"]["p"])
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

        if "thin_plate_spline" in cfg:    
            if cfg["thin_plate_spline"]["enabled"]:
                transforms.append(
                    K.RandomThinPlateSpline(
                        scale=cfg["thin_plate_spline"]["scale"],
                        p=cfg["thin_plate_spline"]["p"]
                    )
                )
        
        if "random_perspective" in cfg:
            if cfg["random_perspective"]["enabled"]:
                transforms.append(
                    K.RandomPerspective(
                        distortion_scale=cfg["random_perspective"]["distortion_scale"],
                        p=cfg["random_perspective"]["p"]
                    )
                )
            
        # Add random sharpness if enabled
        if "random_sharpness" in cfg:
            if cfg["random_sharpness"]["enabled"]:
                transforms.append(
                    K.RandomSharpness(
                        sharpness=cfg["random_sharpness"]["sharpness"],
                        p=cfg["random_sharpness"]["p"]
                    )
                )
            
        # Add random posterize if enabled
        if "random_posterize" in cfg:
            if cfg["random_posterize"]["enabled"]:
                transforms.append(
                    K.RandomPosterize(
                        bits=cfg["random_posterize"]["bits"],
                        p=cfg["random_posterize"]["p"]
                    )
                )
            
        if "random_erasing" in cfg:
            if cfg["random_erasing"]["enabled"]:
                transforms.append(
                    K.RandomErasing(
                        scale=tuple(cfg["random_erasing"]["scale"]),
                        p=cfg["random_erasing"]["p"]
                    )
                )   
        
        # Add normalization if enabled
        mean = cfg.get("normalize", {}).get("mean", [0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
        std = cfg.get("normalize", {}).get("std", [0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
        
        # If RGB-only mode is enabled, use only the first 3 values
        if self.rgb_only and len(mean) > 3:
            mean = mean[:3]
            std = std[:3]
        # If UV-only mode is enabled, use only the last 3 values
        elif self.uv_only and len(mean) > 3:
            mean = mean[3:6]  # UV channels are typically the last 3
            std = std[3:6]
        # If the provided mean/std are for RGB (3 channels), duplicate them for 6 channels
        elif not self.rgb_only and not self.uv_only and len(mean) == 3:
            mean = mean * 2
            std = std * 2
        elif self.usml:
            mean = mean[:5]
            std = std[:5]
            
        # transforms.append(
        #     K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        # )
        
        # Create a sequential transform and wrap it in KorniaSpectralTransform
        return KorniaTransform(torch.nn.Sequential(*transforms))

    # Define a class method for the transform function
    def apply_predict_transform(self, x):
        """Apply the prediction transform to a tensor."""
        if self.predict_transform is None:
            raise ValueError("Predict transform is not initialized. Call setup('predict') first.")
        return self.predict_transform(x.unsqueeze(0)).squeeze(0)



    def setup(self, stage: str = "train"):

        if stage == "fit":
            self.train_dataset = MultispectralTifDataset(
                input_dir=self.data_dir,
                transform=self.train_transform,
                rgb_only=self.rgb_only,
                uv_only=self.uv_only,
                usml=self.usml,
                channels=self.channels
            )

        if stage == "predict":
            # For prediction, we only need to resize and normalize
            # Create a simple transform pipeline for prediction
            normalize_cfg = self.config.get("augmentations", {}).get("transforms", {}).get("normalize", {})
            
            # Get normalization parameters with defaults
            mean = normalize_cfg.get("mean", [0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
            std = normalize_cfg.get("std", [0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
            
            # If RGB-only mode is enabled, use only the first 3 values
            if self.rgb_only and len(mean) > 3:
                mean = mean[:3]
                std = std[:3]
            # If UV-only mode is enabled, use only the last 3 values
            elif self.uv_only and len(mean) > 3:
                mean = mean[3:6]  # UV channels are typically the last 3
                std = std[3:6]
            # If the provided mean/std are for RGB (3 channels), duplicate them for 6 channels
            elif not self.rgb_only and not self.uv_only and len(mean) == 3:
                mean = mean * 2
                std = std * 2
            elif self.usml:
                mean = mean[:5]
                std = std[:5]
            
            self.predict_transform = K.AugmentationSequential(
                K.Resize(size=(self.input_size, self.input_size)),
                # K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
                data_keys=["input"]
            )
            
            # Use the custom transform class instead of a lambda
            predict_transform_fn = PredictTransform(self.predict_transform)
            
            self.predict_dataset = MultispectralTifDataset(
                input_dir=self.data_dir,
                transform=predict_transform_fn,
                rgb_only=self.rgb_only,
                uv_only=self.uv_only,
                usml=self.usml,
                channels=self.channels
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )


    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use single process to avoid pickling issues
            drop_last=False,
            pin_memory=self.pin_memory,
            prefetch_factor=None  # Not used when num_workers=0
        )
    

def custom_collate(batch):
    """
    Custom collate function to handle potential errors in the batch.
    
    Args:
        batch: A batch of data
        
    Returns:
        Collated batch or None if no valid items
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    print(f"Batch size after filtering: {len(batch)}")

    if len(batch) == 0:
        return None
    
    # Check if batch items include filenames
    if len(batch[0]) == 3:
        # Format: (image, label, filename)
        images = []
        labels = []
        filenames = []
        
        for item in batch:
            try:
                image, label, filename = item
                images.append(image)
                labels.append(label)
                filenames.append(filename)
            except Exception as e:
                print(f"Warning: Skipping problematic item: {e}")
                continue
        
        # If no valid items, return None
        if len(images) == 0:
            return None
            
        # Stack tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels, filenames
    else:
        # Format: (image, label)
        images = []
        labels = []
        
        for item in batch:
            try:
                image, label = item
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping problematic item: {e}")
                continue
        
        # If no valid items, return None
        if len(images) == 0:
            return None
            
        # Stack tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels


class SimCLRCollate:
    """
    Custom collate function specifically for SimCLR, which needs to handle pairs of augmented views.
    Unlike the standard collate function, this expects a special transform that returns pairs.
    """
    def __call__(self, batch):
        # Filter out None values
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None
        
        # For SimCLR, we expect (img, label, filename)
        images = []
        labels = []
        filenames = []
        
        for item in batch:
            try:
                img, label, filename = item
                images.append(img)
                labels.append(label)
                filenames.append(filename)
            except Exception as e:
                print(f"Warning: Skipping problematic item in SimCLRCollate: {e}")
                continue
        
        if len(images) == 0:
            return None
        
        # No need to stack here since the transform will be applied later
        return images, labels, filenames


class KorniaSimCLRTransform:
    """
    Applies Kornia transforms to create two views of each image for contrastive learning.
    """
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, images):
        # Apply transform to each image to get two augmented views
        aug_pairs = []
        for img in images:
            # Create two views of the same image
            aug1 = self.transform(img)
            aug2 = self.transform(img)
            
            # Check if the transforms returned tuples
            if isinstance(aug1, tuple):
                aug1 = aug1[0]
            if isinstance(aug2, tuple):
                aug2 = aug2[0]
            
            aug_pairs.append((aug1, aug2))
        
        # Separate the pairs into two lists
        aug1s, aug2s = zip(*aug_pairs)
        
        # Stack the tensors
        aug1s = torch.stack(aug1s)
        aug2s = torch.stack(aug2s)
        
        return (aug1s, aug2s)