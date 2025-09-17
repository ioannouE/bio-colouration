import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import datetime as dt
import json
import os
import re
import yaml
import glob
from typing import Dict, Any, List, Tuple, Optional, Union

import sys # Add sys import
# Dynamically add the 'lepidoptera' directory to sys.path to import 'read_iml_hyp'
# __file__ is the path to the current script.
# os.path.dirname(__file__) is the directory of the current script (simclr/scripts/)
# os.path.join(..., '..', 'lepidoptera') navigates to simclr/lepidoptera/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_lepidoptera_dir = os.path.abspath(os.path.join(_script_dir, '..', 'lepidoptera'))
if _lepidoptera_dir not in sys.path:
    sys.path.insert(0, _lepidoptera_dir) # Prepend to path to prioritize this version

try:
    from read_iml_hyp import read_iml_hyp
except ImportError as e:
    print(f"ERROR: Could not import 'read_iml_hyp' from '{_lepidoptera_dir}'. "
          f"Please ensure 'read_iml_hyp.py' exists in that directory and it's a Python module. Details: {e}")
    # Depending on the application's needs, you might re-raise the error or exit.
    # For now, we'll print the error and let the program continue,
    # which will likely lead to a NameError later if read_iml_hyp is called.
    # To make the import failure fatal, uncomment the next line:
    # raise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import rasterio
from rasterio.plot import reshape_as_image

from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from lightning.pytorch.loggers import TensorBoardLogger

# Import Kornia augmentations
import kornia.augmentation as K
from kornia.constants import BorderType
from kornia.geometry.transform import get_perspective_transform, warp_perspective
import kornia 

from hyperspectral_data import HyperspectralDataModule, HyperspectralBilDataset
import pytz

import tempfile
# Set temporary directory on same filesystem as checkpoint destination
tmpdir = "/shared/cooney_lab/Shared/Eleftherios-Ioannou/tmp"
os.environ["TMPDIR"] = tmpdir
os.environ["FSSPEC_TEMP"] = tmpdir
tempfile.tempdir = tmpdir



class SpectralAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_dim = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Global Average Pooling → (B, C)
        avg_pool = nn.functional.adaptive_avg_pool2d(x, 1).view(B, C)

        # MLP → Attention weights
        attn = self.fc1(avg_pool)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn).view(B, C, 1, 1)

        # Reweight original input
        return x * attn


def modify_vit_input_channels(model: nn.Module, in_channels: int):
    """Modify ViT patch embedding to accept in_channels instead of RGB."""
    patch_embed = model.conv_proj  # Conv2d(3, hidden_dim, kernel_size=16, stride=16)
    new_patch_embed = nn.Conv2d(in_channels, patch_embed.out_channels,
                                kernel_size=patch_embed.kernel_size,
                                stride=patch_embed.stride,
                                padding=patch_embed.padding,
                                bias=patch_embed.bias is not None)
    
    # Optional: initialize new weights with pretrained RGB weights
    # if in_channels >= 3:
    #     with torch.no_grad():
    #         new_patch_embed.weight[:, :3, :, :] = patch_embed.weight  # copy RGB weights
    #         if in_channels > 3:
    #             # nn.init.kaiming_normal_(new_patch_embed.weight[:, 3:, :, :])  # init extra channels
    #             # do orthogonal initialization
    #             nn.init.orthogonal_(new_patch_embed.weight[:, 3:, :, :])
    # else:
        # nn.init.kaiming_normal_(new_patch_embed.weight)
        # do orthogonal initialization
    # nn.init.orthogonal_(new_patch_embed.weight)
    
    model.conv_proj = new_patch_embed
    return model


class MultispectralBackbone(nn.Module):
    """Custom backbone for multispectral images."""
    
    def __init__(self, base_model, in_channels=6, checkpoint_path=None):
        super().__init__()
        
        # Adapter to convert input channels to RGB (3 channels)
        # self.adapter = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 3, kernel_size=1)
        )

        # Use the entire ViT model except the head
        self.backbone = base_model
        
        # Initialize adapter with Kaiming initialization
        # print("Initializing adapter with Kaiming initialization")
        # nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.xavier_uniform_(self.adapter.weight)
        # nn.init.orthogonal_(self.adapter.weight)

        if checkpoint_path:
            # Load weights from MultispectralClassifier checkpoint
            print(f"Loading pretrained weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                # Load adapter weights
                adapter_key = 'model.adapter.weight'
                if adapter_key in state_dict:
                    self.adapter.weight.data = state_dict[adapter_key]
                    print("Successfully loaded adapter weights")
                
                # Load ViT weights if they exist in checkpoint
                loaded_params = 0
                total_params = 0
                for name, param in self.backbone.named_parameters():
                    total_params += 1
                    checkpoint_key = f'model.original_model.{name}'
                    if checkpoint_key in state_dict:
                        param.data = state_dict[checkpoint_key]
                        loaded_params += 1
                
                print(f"Successfully loaded backbone weights ({loaded_params}/{total_params} parameters)")
    
    def forward(self, x):
        # Apply adapter to convert input channels to RGB
        x = self.adapter(x)
        
        # Ensure the input is in the correct format for ViT
        # ViT expects input shape: (batch_size, channels, height, width)
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Make tensor contiguous and ensure correct shape
        x = x.contiguous()
        
        # Forward through ViT backbone
        x = self.backbone(x)
        
        # ViT output is [batch_size, hidden_dim]
        return x

class SimCLRModel(pl.LightningModule):
    def __init__(self, config: dict, lr: float = 6e-2, T_max: int = 5, in_channels: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.T_max = config.get('scheduler').get('params', {}).get('T_max', T_max)
        self.lr = config.get('optimizer').get('lr', lr)
        self.temp = config.get('criterion').get('params', {}).get('temperature', 0.5)
        # set SA to None and adjust later
        self.spectral_attn = None 
        
        # Get loss configuration
        criterion_config = config.get('criterion', {})
        self.loss_type = criterion_config.get('type', 'ntxent')
        self.global_weight = criterion_config.get('global_weight', 1.0)  
        self.local_weight = criterion_config.get('local_weight', 0.0)
           
        print(f"Using loss={self.loss_type} with weights: Global NTXent={self.global_weight}, Local NTXent={self.local_weight}")

        # Flag to determine if we should use local patch-level loss
        self.use_local_loss = criterion_config.get('use_local_loss', False)  
        print(f"Using local loss: {self.use_local_loss}")

        # create a backbone and remove the classification head
        model = config.get("backbone")
        weights = config.get('weights')
        # base_model = torchvision.models.get_model(model, weights=weights)

        if model == 'resnet50':
            base_model = torchvision.models.get_model(model, weights=weights)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            hidden_dim = base_model.fc.in_features
            self.is_vit = False
        else:
            # Import ViT models first
            from torchvision.models import vit_l_16, ViT_L_16_Weights

            # Create ViT backbone with explicit image size and patch size
            weights = ViT_L_16_Weights.IMAGENET1K_V1
            base_model = vit_l_16(weights=weights, image_size=224)

            # Remove classification head
            base_model.heads = nn.Identity()

            # ViT-L/16 has hidden dimension of 1024 (larger than ViT-B/16's 768)
            hidden_dim = 1024
            self.is_vit = True
            # Store patch size for later use in extracting patch tokens
            self.patch_size = 16

        if in_channels > 3:
            checkpoint_path = config.get('checkpoint_path', None)
            if model == 'resnet50':
                self.backbone = MultispectralBackbone(base_model, in_channels=in_channels, checkpoint_path=checkpoint_path)
            else:
                # self.spectral_attn = SpectralAttention(in_channels=in_channels)
                base_model = modify_vit_input_channels(base_model, in_channels=in_channels)
                self.backbone = base_model
        else:
            self.backbone = base_model
            
        

        # Define a custom projection head to handle contiguity issues
        # self.projection_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 128)
        # )

        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        
        # Add local projection head for patch-level features (lighter than global head)
        if self.is_vit and self.use_local_loss:
            self.local_projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 128)
            )
        
        # Initialize loss function - only need NTXentLoss now
        self.ntxent_criterion = NTXentLoss(self.temp)

    def _extract_patch_tokens(self, x):
        """Extract patch tokens from ViT model before class token aggregation"""
        if not self.is_vit:
            return None
            
        # Access the internal ViT methods to get patch tokens
        # This follows the same process as in VisionTransformer._process_input
        n, c, h, w = x.shape
        p = self.patch_size
        
        # Process input through conv_proj
        x = self.backbone.conv_proj(x)
        # Reshape to get patch tokens
        n_h = h // p
        n_w = w // p
        x = x.reshape(n, self.backbone.hidden_dim, n_h * n_w)
        # Permute to get shape [batch_size, num_patches, hidden_dim]
        x = x.permute(0, 2, 1)
        
        # Process through encoder but return all tokens, not just class token
        # Add class token
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Apply position embeddings and encoder
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)
        x = self.backbone.encoder.layers(x)
        x = self.backbone.encoder.ln(x)
        
        # Return all tokens except class token [batch_size, num_patches, hidden_dim]
        patch_tokens = x[:, 1:, :]
        
        return patch_tokens

    def forward(self, x):
        # Ensure input tensor is contiguous and in correct shape
        x = x.contiguous()
        if self.spectral_attn is not None:
            x = self.spectral_attn(x)
        
        # Extract patch tokens if using ViT and local loss is enabled
        patch_tokens = None
        if self.is_vit and self.use_local_loss:
            patch_tokens = self._extract_patch_tokens(x)
        
        # Get embeddings from backbone
        h = self.backbone(x)  # Output shape: [batch_size, hidden_dim]
        
        # Ensure h is contiguous before projection
        h = h.contiguous()
        
        # Project embeddings
        z = self.projection_head(h)
        
        # If we have patch tokens, project them too
        local_z = None
        if patch_tokens is not None:
            # Project each patch token
            # patch_tokens shape: [batch_size, num_patches, hidden_dim]
            local_z = self.local_projection_head(patch_tokens)
            
        return z, local_z

    def compute_local_loss(self, local_z1, local_z2):
        """
        Compute NTXent loss at the patch level
        
        Args:
            local_z1: Patch features from first view [batch_size, num_patches, proj_dim]
            local_z2: Patch features from second view [batch_size, num_patches, proj_dim]
            
        Returns:
            Local NTXent loss
        """
        batch_size, num_patches, proj_dim = local_z1.shape
        
        # Initialize loss
        local_loss = 0.0
        
        # For each patch position, compute NTXent loss
        for i in range(num_patches):
            # Extract features for the i-th patch across all batch samples
            p1_i = local_z1[:, i, :]  # [batch_size, proj_dim]
            p2_i = local_z2[:, i, :]  # [batch_size, proj_dim]
            
            # Compute NTXent loss for this patch position
            patch_loss = self.ntxent_criterion(p1_i, p2_i)
            local_loss += patch_loss
            
        # Average over all patch positions
        local_loss = local_loss / num_patches

        return local_loss

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        
        # Ensure both views are contiguous
        x0 = x0.contiguous()
        x1 = x1.contiguous()
        
        # Forward pass through the model
        z0, local_z0 = self.forward(x0)
        z1, local_z1 = self.forward(x1)
        
        # Compute global NTXent loss
        global_loss = self.ntxent_criterion(z0, z1)
        self.log("train_global_loss", global_loss)
        
        # Initialize total loss with global contribution
        total_loss = self.global_weight * global_loss
        
        # Compute local patch-level loss if enabled and weight > 0
        if self.local_weight > 0 and self.use_local_loss and local_z0 is not None and local_z1 is not None:
            local_loss = self.compute_local_loss(local_z0, local_z1)
            self.log("train_local_loss", local_loss)
            total_loss += self.local_weight * local_loss
            
        self.log("train_loss_ssl", total_loss)
        return total_loss

    def configure_optimizers(self):
        # Initialize optimizer with model parameters
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.T_max)
        return [optim], [scheduler]
        # return {
        #     "optimizer": optim,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }


def generate_embeddings(model, dataloader):
    """Generate embeddings for all images in the dataloader."""
    model.eval()
    embeddings = []
    filenames = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch - now returns (tensor, filename)
            img_pair, batch_filenames = batch
            
            # Take only the first view of each pair
            if isinstance(img_pair, tuple):
                img = img_pair[0]  # Take first augmented view
            else:
                img = img_pair  # No augmentation
                
            # Move to device
            img = img.to(model.device)
            
            # Get embeddings using only the backbone
            embedding = model.backbone(img)
            
            # For ViT, the output is already flattened, no need for flatten
            embeddings.append(embedding.cpu())
            filenames.extend(batch_filenames)
    
    # Concatenate all embeddings
    embeddings = torch.cat(embeddings, dim=0)
    
    # Normalize embeddings for cosine similarity
    # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings, filenames



def get_image_as_np_array(filename: str):
    """Returns an image as a numpy array"""
    # Try to load file with PIL for common formats
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        try:
            img = Image.open(filename)
            # Convert to numpy array
            # Ensure image is in a format that can be easily converted (e.g., RGB, L)
            if img.mode not in ['RGB', 'L', 'RGBA']:
                img = img.convert('RGB') # Convert to RGB if complex mode
            return np.array(img)
        except Exception as e:
            print(f"Error opening image file {filename} with PIL: {e}")
            return None
    
    # Handle .bil files (hyperspectral)
    elif filename.lower().endswith(".bil"):
        try:
            # Attempt to use spectral library if available
            from spectral import open_image
            hdr_file = filename + '.hdr'
            if not os.path.exists(hdr_file):
                hdr_file = filename[:-4] + '.hdr' # Try base name + .hdr

            if os.path.exists(hdr_file):
                img_spectral = open_image(hdr_file)
                # Ensure we get a numpy array; .load() might return the object itself
                # or specific bands. Reading all bands as a numpy array is safest.
                img_data = img_spectral.read_bands(range(img_spectral.nbands))
                # The shape might be (bands, lines, samples) or (lines, samples, bands)
                # We typically want (lines, samples, bands) or (H, W, C)
                if img_data.shape[0] == img_spectral.nbands and img_data.ndim == 3:
                     # if bands first, transpose to bands last (lines, samples, bands)
                    img_data = np.transpose(img_data, (1, 2, 0))
                return img_data
            else:
                print(f"Warning: BIL file {filename} found, but no corresponding .hdr file. Falling back to read_iml_hyp.")
                directory = os.path.dirname(filename)
                base_name = os.path.basename(filename)[:-4] # Remove .bil
                img_data, _, _, _ = read_iml_hyp(directory, base_name) # Unpack tuple
                return img_data # Return only the image array

        except ImportError:
            print("INFO: 'spectral' module not found. Falling back to custom BIL reader (read_iml_hyp) for .bil files.")
            directory = os.path.dirname(filename)
            base_name = os.path.basename(filename)[:-4] # Remove .bil
            img_data, _, _, _ = read_iml_hyp(directory, base_name) # Unpack tuple
            return img_data # Return only the image array
        except Exception as e:
            print(f"Error processing BIL file {filename} with spectral library or custom reader: {e}")
            # Attempt one last fallback to read_iml_hyp if spectral processing failed for other reasons
            try:
                print(f"Attempting final fallback to read_iml_hyp for {filename}")
                directory = os.path.dirname(filename)
                base_name = os.path.basename(filename)[:-4]
                img_data, _, _, _ = read_iml_hyp(directory, base_name) # Unpack tuple
                return img_data # Return only the image array
            except Exception as fallback_e:
                print(f"Final fallback with read_iml_hyp also failed for {filename}: {fallback_e}")
                return None

    # Handle other files by assuming they are base paths for read_iml_hyp
    else:
        # print(f"Attempting to load non-standard file as hyperspectral (base path): {filename}")
        try:
            directory = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            # read_iml_hyp returns a 4-tuple: (image_array, wavelengths, scan_info, gain)
            # We only need the image_array.
            img_data, _, _, _ = read_iml_hyp(directory, base_name) # Unpack the tuple
            return img_data # Return only the image array
        except Exception as e:
            # This is where the user's error message "Error loading file ...: expected np.ndarray (got tuple)" was printed from
            # If 'e' itself is that TypeError, it means read_iml_hyp or a function it calls raised it.
            # The unpacking above should prevent get_image_as_np_array from *returning* a tuple if read_iml_hyp succeeds.
            print(f"Error loading file {filename} using read_iml_hyp (likely as base path): {e}")
            return None
    
    # Fallback if no other condition met (should ideally not be reached)
    print(f"Warning: Could not determine how to load file: {filename}")
    return None


# Plots multiple rows of random images with their nearest neighbors
def plot_knn_examples(embeddings, filenames, data_dir, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    
    fig, axs = plt.subplots(num_examples, n_neighbors, figsize=(12, 2*num_examples))
    # loop through our randomly picked samples
    for plot_y_offset, idx in enumerate(samples_idx):
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = axs[plot_y_offset, plot_x_offset]
            # get the corresponding filename for the current index
            fname = os.path.join(data_dir, filenames[neighbor_idx])
            
            # Debug output to trace what we're trying to load
            print(f"Loading KNN image: {fname}")
            
            try:
                # Get and plot the image
                img_data = get_image_as_np_array(fname) # Renamed from img_array
                
                # Prepare image for display (e.g., select bands for hyperspectral)
                processed_img_for_display = None
                if img_data is not None and img_data.size > 0:
                    if img_data.ndim == 3 and img_data.shape[2] > 4: # Likely hyperspectral
                        num_bands = img_data.shape[2]
                        # Select three bands (e.g., R, G, B like). These are example percentages.
                        # Adjust if specific wavelength mappings are known for better color representation.
                        b_idx = min(num_bands - 1, int(num_bands * 0.3))
                        g_idx = min(num_bands - 1, int(num_bands * 0.5))
                        r_idx = min(num_bands - 1, int(num_bands * 0.7))
                        processed_img_for_display = img_data[:, :, [r_idx, g_idx, b_idx]]
                        # print(f"INFO: Hyperspectral {fname} (shape {img_data.shape}), displaying bands [{r_idx},{g_idx},{r_idx}]")
                    elif img_data.ndim == 3 and img_data.shape[2] == 1: # Grayscale image in 3D tensor (H, W, 1)
                        processed_img_for_display = img_data[:, :, 0] # Convert to 2D for grayscale display
                    else: # Assumed to be 2D (grayscale) or 3D (RGB, H, W, 3) or other displayable formats
                        processed_img_for_display = img_data
                else: # img_data is None or empty
                    print(f"Warning: Image {fname} is None or empty after loading. Using placeholder.")
                    processed_img_for_display = np.zeros((50, 50, 3), dtype=np.uint8) # Small black placeholder
                
                # Normalize the processed image for display
                if processed_img_for_display.dtype != np.uint8 and processed_img_for_display.size > 0:
                    max_val = np.max(processed_img_for_display) # Use np.max for numpy arrays
                    if max_val > 0:
                        processed_img_for_display = (processed_img_for_display / max_val * 255).astype(np.uint8)
                    else:
                        processed_img_for_display = np.zeros_like(processed_img_for_display, dtype=np.uint8)
                
                ax.imshow(processed_img_for_display)
            except Exception as e:
                print(f"Failed to display image {fname}: {e}")
                # Display error placeholder 
                ax.text(0.5, 0.5, "Image load error", ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('lightgray')
                
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            ax.axis("off")
    
    plt.tight_layout()
    return fig


def get_args():
    """
    Parse command line arguments
    ---------
    Returns:
        argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration file",
        default="./config.yaml",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Use only RGB channels (first 3 layers) and ignore UV data",
    )
    parser.add_argument(
        "--sample-bands",
        type=str,
        help="Comma-separated list of band indices to sample from hyperspectral data (e.g., '0,50,100,150,200')",
        default=None
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        help="Path to pretrained MultispectralClassifier checkpoint to use as backbone",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize augmentations applied to random samples from the dataset",
    )
    # Add new arguments for loss function
    parser.add_argument(
        "--loss-type",
        type=str,
        default="global",
        choices=["global", "local", "combined"],
        help="Loss function to use (global, local, or combined)",
    )
    parser.add_argument(
        "--global-weight",
        type=float,
        default=1.0,
        help="Weight for global NTXentLoss",
    )
    parser.add_argument(
        "--local-weight",
        type=float,
        default=1.0,
        help="Weight for local NTXentLoss",
    )
    parser.add_argument(
        "--test-embeddings-only",
        action="store_true",
        help="Skip training and only generate embeddings for testing",
    )
    parser.add_argument(
        "--augmentation-strategy",
        type=str,
        default="crop",
        choices=["crop", "resize"],
        help="Augmentation strategy to use (crop or resize)",
    )
    return parser.parse_args()

def test_model_with_dummy_input(model, in_channels=400, input_size=224, device=None):
    """
    Test the model with a dummy input to ensure everything is working
    ----------
    Args:
        model: torch.nn.Module
            Model to test
        in_channels: int
            Number of input channels
        input_size: int
            Input image size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model with dummy input (channels={in_channels}, size={input_size}x{input_size})...")
    
    # Create dummy input
    dummy_input = torch.randn(1, in_channels, input_size, input_size)
    dummy_input = dummy_input.to(device)
    
    # Set model to eval mode
    model.eval()
    
    try:
        # Forward pass
        with torch.no_grad():
            # Test full model
            output = model(dummy_input)
            print(f"Full model output shape: {output.shape}")
            print(f"Output contains NaN: {torch.isnan(output).any().item()}")
        
        print("Model test successful!")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False


def visualize_augmentations(datamodule, num_samples=5, save_dir="augmentation_examples"):
    """
    Visualize augmentations applied to images from the dataset
    
    Parameters:
    -----------
    datamodule : HyperspectralDataModule
        The data module containing the dataset
    num_samples : int
        Number of samples to visualize
    save_dir : str
        Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup the datamodule
    datamodule.setup()
    
    # Get samples from the training dataset
    dataset = datamodule.train_dataset
    
    # For hyperspectral data, we'll use a simple approach to convert to RGB for visualization
    def convert_to_rgb(tensor):
        """Convert hyperspectral tensor to RGB for visualization"""
        if tensor.shape[0] == 3:  # Already RGB format
            return tensor
        
        # Get total number of channels
        channels = tensor.shape[0]
        
        # Use simple channel selection: RED=channels//6, GREEN=channels//2, BLUE=5*channels//6
        # This is a very simplistic conversion just for visualization purposes
        red_idx = min(channels // 6, channels - 1)
        green_idx = min(channels // 2, channels - 1)
        blue_idx = min(5 * channels // 6, channels - 1)
        
        rgb = torch.stack([
            tensor[red_idx],
            tensor[green_idx],
            tensor[blue_idx]
        ], dim=0)
        
        # Normalize to [0, 1] for display
        for i in range(3):
            if rgb[i].max() != rgb[i].min():
                rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())
            else:
                rgb[i] = torch.zeros_like(rgb[i])
        
        return rgb
    
    print(f"Generating augmentation examples for {num_samples} random samples...")
    
    # Select random samples
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    for i, idx in enumerate(tqdm(indices)):
        # Get a sample from dataset
        sample = dataset[idx]
        
        # Unpack data and augmentations based on expected format
        if isinstance(sample, tuple) and len(sample) == 2:
            # Format is either ((view1, view2), filename) or (tensor, filename)
            first_element, filename = sample
            
            if isinstance(first_element, tuple) and len(first_element) == 2:
                # This means we have SimCLR views
                view1, view2 = first_element
                
                # Convert to RGB for visualization
                view1_rgb = convert_to_rgb(view1)
                view2_rgb = convert_to_rgb(view2)
                
                # Create a figure with 3 plots: original (we don't have), view1, view2
                plt.figure(figsize=(12, 4))
                
                # Plot view 1
                plt.subplot(1, 2, 1)
                plt.title(f"Augmentation 1")
                plt.imshow(view1_rgb.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                
                # Plot view 2
                plt.subplot(1, 2, 2)
                plt.title(f"Augmentation 2")
                plt.imshow(view2_rgb.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                
                # Save figure
                file_id = os.path.splitext(os.path.basename(filename))[0]
                save_path = os.path.join(save_dir, f"aug_sample_{file_id}.png")
                plt.suptitle(f"Sample: {file_id}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                # This is just a regular transformed tensor with filename
                print(f"Skipping sample {i} - not in SimCLR format")
        else:
            print(f"Skipping sample {i} - unexpected format")
    
    print(f"Augmentation examples saved to {save_dir}/")


def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    # Process sample_bands if provided
    sample_bands = None
    if args.sample_bands:
        try:
            sample_bands = [int(band) for band in args.sample_bands.split(',')]
            print(f"Using specific bands: {sample_bands} ({len(sample_bands)} channels)")
            config["data"]["sample_bands"] = sample_bands
            in_channels = len(sample_bands)
        except ValueError:
            print("Error parsing sample bands. Make sure it's a comma-separated list of integers.")
            raise
    
    # Add rgb_only flag to config if specified
    elif args.rgb_only:
        print("RGB-like mode enabled: Using only bands closest to RGB wavelengths (3 channels)")
        config["data"]["rgb_only"] = True
        config["model"]["rgb_only"] = True
        in_channels = 3
    else:
        print("Using all hyperspectral bands")
        # Instead of hardcoding a value, we'll create a temporary dataset to get the actual number of channels
        # This ensures the model is initialized with the correct number of input channels
        temp_dataset = HyperspectralBilDataset(
            input_dir=config["data"]["data_dir"],
            transform=None,
            rgb_only=False,
            normalize=True
        )
        # Get the first item to check its dimensions
        try:
            first_item, _ = temp_dataset[0]
            actual_channels = first_item.shape[0]
            print(f"Detected {actual_channels} channels from the hyperspectral dataset")
            in_channels = actual_channels
        except Exception as e:
            print(f"Could not determine channels from dataset: {e}")
            print("Falling back to default of 408 channels based on previous error message")
            in_channels = 408  # Using 408 as default based on the error message
    
    # Add model checkpoint path to config if specified
    if args.model_checkpoint:
        print(f"Using pretrained MultispectralClassifier weights from: {args.model_checkpoint}")
        config["model"]["checkpoint_path"] = args.model_checkpoint

    # Set loss configuration in config
    if 'criterion' not in config:
        config['criterion'] = {}
    config['criterion']['type'] = args.loss_type
    
    # Add weights for combined loss
    if args.loss_type == 'combined':
        print(f"Using combined loss with weights: Global NTXent={args.global_weight}, Local NTXent={args.local_weight}")
        config['criterion']['global_weight'] = args.global_weight
        config['criterion']['local_weight'] = args.local_weight
    else:
        print(f"Using {args.loss_type} loss")

    # Trade-off precision for performance when using CUDA device that has Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Let's set the seed for our experiments
    pl.seed_everything(config["seed"])

    data_dir = config["data"]["data_dir"]

    # Extract the name of the config file without the directory path or extension
    config_filename = os.path.splitext(os.path.basename(args.config))[0]

    # Create a timestamped version string that includes the config filename
    # uk_tz = pytz.timezone('Europe/London')
    # date_str = dt.datetime.now(uk_tz).astimezone(pytz.utc).isoformat(timespec='minutes')
    # date_str = dt.datetime.isoformat(dt.datetime.utcnow(), timespec='minutes')
    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec='minutes')
    version = f"{config_filename}_{''.join(re.split(r'[-:]', date_str))}"
    
    # Set up logger
    logger = TensorBoardLogger(config["output"]["out_dir"],
                               name=config["model"]["name"],
                               version=version)
    
    # ---- START OF QUICK TEST FOR plot_knn_examples ----
    # Set this to True to run this isolated test, False to run the normal script
    RUN_ISOLATED_PLOT_KNN_TEST = False 

    if RUN_ISOLATED_PLOT_KNN_TEST and __name__ == "__main__":
        print("--- RUNNING ISOLATED QUICK TEST FOR plot_knn_examples ---")
        
        current_data_dir_for_test = ""
        try:
            # Attempt to get data_dir from the loaded config
            current_data_dir_for_test = config["data"]["data_dir"]
        except KeyError:
            print(f"ERROR: 'data_dir' key not found in config['data']. Current config['data'] is: {config.get('data', 'Not Found')}")
            print("Please ensure your config file has data: data_dir: /path/to/your/images")
            if 'sys' in globals() or 'sys' in locals(): sys.exit(1)
            else: os._exit(1) # Fallback if sys is not available
        except TypeError:
            print(f"ERROR: 'config' or 'config['data']' is not as expected. 'config' is: {config}")
            if 'sys' in globals() or 'sys' in locals(): sys.exit(1)
            else: os._exit(1)
        
        if not os.path.isdir(current_data_dir_for_test):
            print(f"ERROR: The data_dir for test '{current_data_dir_for_test}' does not exist or is not a directory.")
            if 'sys' in globals() or 'sys' in locals(): sys.exit(1)
            else: os._exit(1)
        
        mock_num_samples_test = 2 # Number of mock embeddings and filenames
        mock_embedding_dim_test = 128  # Example embedding dimension
        mock_embeddings_test = np.random.rand(mock_num_samples_test, mock_embedding_dim_test)
        
        # !!! IMPORTANT: Replace these with actual filenames (without directory) 
        # that exist in your 'current_data_dir_for_test'.
        # Ensure at least one is a hyperspectral image like 'Pieris rapae_2D_030717'.
        mock_filenames_test = [
            "Pieris rapae_2D_030717",  # EXAMPLE - REPLACE IF NEEDED
            # "another_actual_image_name.jpg", # EXAMPLE - REPLACE OR REMOVE
        ]
        # Ensure the list has mock_num_samples_test elements
        if len(mock_filenames_test) < mock_num_samples_test:
            print(f"Warning: Test needs {mock_num_samples_test} filenames, got {len(mock_filenames_test)}. Adjusting or using duplicates.")
            if not mock_filenames_test: # if list is empty, add a placeholder
                mock_filenames_test = ["PLEASE_REPLACE_ME_WITH_VALID_FILENAME"] * mock_num_samples_test
            else:
                mock_filenames_test.extend([mock_filenames_test[0]] * (mock_num_samples_test - len(mock_filenames_test)))
        elif len(mock_filenames_test) > mock_num_samples_test:
            mock_filenames_test = mock_filenames_test[:mock_num_samples_test]
        
        print(f"Test - Embeddings shape: {mock_embeddings_test.shape}")
        print(f"Test - Filenames: {mock_filenames_test}")
        print(f"Test - Using data_dir: {current_data_dir_for_test}")
        
        # Ensure n_neighbors <= number of samples in NearestNeighbors
        test_n_neighbors = min(2, mock_num_samples_test) 
        test_num_examples = min(2, mock_num_samples_test)
        
        print(f"Test - Calling plot_knn_examples with n_neighbors={test_n_neighbors}, num_examples={test_num_examples}")
        try:
            fig = plot_knn_examples(mock_embeddings_test, mock_filenames_test, current_data_dir_for_test, 
                                    n_neighbors=test_n_neighbors, num_examples=test_num_examples)
            if fig is not None:
                plt.suptitle("QUICK TEST (ISOLATED): plot_knn_examples", fontsize=10)
                plt.tight_layout()
                plt.savefig("quick_test_plot_knn_examples.png")
                plt.show() 
                print("--- ISOLATED QUICK TEST FINISHED. Plot window might need to be closed manually. ---")
            else:
                print("--- ISOLATED QUICK TEST: plot_knn_examples did not return a figure. ---")
        except Exception as e:
            print(f"ERROR during isolated plot_knn_examples test: {e}")
            import traceback
            traceback.print_exc()
        
        print("Exiting script after isolated test.")
        if 'sys' in globals() or 'sys' in locals(): sys.exit(0)
        else: os._exit(0)
    # ---- END OF QUICK TEST FOR plot_knn_examples ----

    if args.visualize:
        print("Generating augmentation visualization...")
        datamodule = HyperspectralDataModule(config)
        visualize_augmentations(datamodule)
        print("Visualization saved in augmentation_examples/")
        return


    # Set up dataloader and model for training
    dm = HyperspectralDataModule(config, rgb_only=args.rgb_only, sample_bands=sample_bands, augmentation_strategy=args.augmentation_strategy)
    model = SimCLRModel(config["model"], in_channels=in_channels)
    
    # Test the model with a dummy input
    # in_channels = 3 if config["data"].get("rgb_only", False) else 6
    # test_model_with_dummy_input(model, in_channels=in_channels)
    
    # Set up trainer with default values if not specified
    trainer_config = config.get("train", {})
    default_trainer_config = {
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": 1,
        "log_every_n_steps": 10
    }
    
    # Merge default config with provided config
    for key, value in default_trainer_config.items():
        if key not in trainer_config:
            trainer_config[key] = value
    
    # Configure checkpointing
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    print(f"Checkpoints will be saved in: {checkpoint_dir}")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{train_loss_ssl:.4f}',
        save_top_k=1,
        save_last=True,
        monitor='train_loss_ssl',
        mode='min'
    )
    
    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=[checkpoint_callback])

    if not args.test_embeddings_only:
        print("Starting training...")
        trainer.fit(model, datamodule=dm)
        print("Training finished.")
    else:
        print("Skipping training, proceeding to generate embeddings.")

    # Generate embeddings
    # Set up dataloader for prediction
    dm.setup(stage="predict")
    model.eval()
    print("Generating embeddings...")
    embeddings, filenames = generate_embeddings(model, dm.predict_dataloader())
    
    # Save embeddings to output dir
    # pd.DataFrame(embeddings, index=filenames).to_csv(os.path.join(logger.log_dir, "embeddings.csv"), quoting=csv.QUOTE_ALL, encoding='utf-8')
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f'x{i+1}' for i in range(len(df_embeddings.columns))]
    df_filenames = pd.DataFrame(filenames, columns=['filename'])
    df = pd.concat([df_filenames, df_embeddings], axis=1)
    output_filename = f"embeddings_{version}.csv"
    embedding_path = os.path.join(logger.log_dir, output_filename)
    print(f"Embeddings will be saved to: {embedding_path}")
    df.to_csv(
        embedding_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        float_format='%.15f')

    print('Embeddings generated...')
    print('Plotting KNN examples...')
    # Create sample plots
    plot_knn_examples(embeddings, filenames, data_dir)

    # Save plots to output dir
    plt.savefig(os.path.join(logger.log_dir, "knn_examples.png"))



if __name__ == "__main__":
    args = get_args()
    main(args)