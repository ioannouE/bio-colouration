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

from multispectral_data import SimCLRDataModule
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
    """Returns an image as an numpy array"""
    # Check if it's a TIF file
    if filename.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(filename) as src:
            # Read all bands
            image = src.read()
            
            # For visualization, use the first 3 bands (visible RGB)
            # The first 3 bands are the visible light channels
            if image.shape[0] >= 3:
                # Get first 3 bands and transpose to (height, width, channels) for display
                visible_rgb = image[:3, :, :].transpose(1, 2, 0)
                # Normalize for better visualization
                visible_rgb = visible_rgb.astype(np.float32)
                if visible_rgb.max() > 0:
                    visible_rgb = visible_rgb / visible_rgb.max() * 255
                return visible_rgb.astype(np.uint8)
            else:
                # If less than 3 bands, transpose and return as is
                return image.transpose(1, 2, 0)
    else:
        # For non-TIF files, use PIL
        img = Image.open(filename)
        return np.asarray(img)



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
            ax = axs[plot_y_offset,
                     plot_x_offset]
            # get the correponding filename for the current index
            fname = os.path.join(data_dir, filenames[neighbor_idx])
            # plot the image
            img_array = get_image_as_np_array(fname)
        
            # Normalize for display if needed
            if img_array.dtype != np.uint8:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            
            ax.imshow(img_array)
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            ax.axis("off")



def visualize_augmentations(datamodule, num_samples=4, num_augmentations=2):
    """Visualize augmentations applied to random samples from the dataset."""
    # Set up the datamodule
    datamodule.setup(stage="fit")
    
    # Get random samples
    dataset = datamodule.train_dataset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Create subplot grid - each row will have: original RGB, original UV, and augmentations
    fig, axes = plt.subplots(num_samples, num_augmentations + 2, figsize=(15, 4*num_samples))
    
    def enhance_image(img):
        """Enhance image visibility"""
        # Convert to float
        img = img.astype(np.float32)
        
        # Enhance contrast using histogram equalization for each channel
        enhanced = np.zeros_like(img)
        for i in range(img.shape[-1]):
            channel = img[..., i]
            p2, p98 = np.percentile(channel, (2, 98))
            if p2 != p98:
                enhanced[..., i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
            else:
                enhanced[..., i] = channel
        
        return enhanced
    
    for i, idx in enumerate(indices):
        # Get original image
        img_path = dataset.filenames[idx]
        
        # Read image using rasterio
        with rasterio.open(img_path) as src:
            img = src.read()
           
            # Handle NaN values
            # img = np.nan_to_num(img, nan=0.0)
            
            # Create RGB visualization (first 3 channels)
            rgb_img = img[:3]
             
            print("Max RGB: ", np.max(rgb_img))
            print("Min RGB: ", np.min(rgb_img))

            # Normalize to [0, 1]
            rgb_img = np.clip(rgb_img / 65535.0, 0, 1)
            # Transpose to HWC format for plotting
            rgb_img = rgb_img.transpose(1, 2, 0)
            # Enhance visibility
            rgb_img = enhance_image(rgb_img)
            
            print("Max RGB: ", np.max(rgb_img))
            print("Min RGB: ", np.min(rgb_img))

            # Create UV visualization (next 3 channels)
            if img.shape[0] >= 6:
                uv_img = img[3:6]
                # Normalize to [0, 1]
                uv_img = np.clip(uv_img / 65535.0, 0, 1)
                # Transpose to HWC format for plotting
                uv_img = uv_img.transpose(1, 2, 0)
                # Enhance visibility
                uv_img = enhance_image(uv_img)
            else:
                uv_img = np.zeros_like(rgb_img)
            
            # Show original RGB
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title('Original (RGB)')
            axes[i, 0].axis('off')
            
            # Show original UV
            axes[i, 1].imshow(uv_img)
            axes[i, 1].set_title('Original (UV)')
            axes[i, 1].axis('off')
        
            # Show augmentations
            for j in range(num_augmentations):
                # Get augmented pair
                aug1, aug2 = dataset[idx][0]

                print("Min aug1: ", torch.min(aug1))
                print("Max aug1: ", torch.max(aug1))
                print("Min aug2: ", torch.min(aug2))
                print("Max aug2: ", torch.max(aug2))
            
                # Select which augmented view to show (alternate between aug1 and aug2)
                aug_img = aug1 if j % 2 == 0 else aug2
                
                aug_img = np.clip(aug_img, 0, 1)
                # Convert from tensor to numpy and adjust dimensions
                aug_img = aug_img.permute(1, 2, 0).numpy()
                
                # Split into RGB and UV components for visualization
                if aug_img.shape[-1] >= 6:
                    aug_rgb = aug_img[..., :3]
                    aug_uv = aug_img[..., 3:6]
                else:
                    aug_rgb = aug_img
                    aug_uv = np.zeros_like(aug_rgb)
                
                # Create a combined visualization
                combined_img = np.concatenate([aug_rgb, aug_uv], axis=1)
                
                combined_img = enhance_image(combined_img)
                
                axes[i, j+2].imshow(combined_img)
                axes[i, j+2].set_title(f'Aug {j+1} (RGB|UV)')
                axes[i, j+2].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('augmentation_examples', exist_ok=True)
    
    # Save the figure
    plt.savefig('augmentation_examples/augmentation_examples.png', 
                bbox_inches='tight', dpi=300)
    plt.close()


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
        "--uv-only",
        action="store_true",
        help="Use only UV channels (last 3 layers) and ignore RGB data",
    )
    parser.add_argument(
        "--usml",
        action="store_true",
        help="Use tetrahedral colour space (USML+DL)",
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
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="List of specific channel indices to use",
    )
    parser.add_argument(
        "--augmentation-strategy",
        type=str,
        default="crop",
        choices=["crop", "resize_and_pad"],
        help="Augmentation strategy to use ('crop' or 'resize_and_pad')",
    )
    return parser.parse_args()

def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    # Add rgb_only flag to config if specified
    if args.rgb_only:
        print("RGB-only mode enabled: Using only the first 3 channels (RGB) and ignoring UV data")
        config["data"]["rgb_only"] = True
        config["model"]["rgb_only"] = True
        in_channels = 3
    elif args.uv_only:
        print("UV-only mode enabled: Using only the last 3 channels (UV) and ignoring RGB data")
        config["data"]["uv_only"] = True
        config["model"]["uv_only"] = True
        in_channels = 3
    elif args.usml:
        print("USML mode enabled: Using tetrahedral colour space (5 channels)")
        config["data"]["usml"] = True
        in_channels = 5
    else:
        # Determine the number of input channels
        if args.channels:
            in_channels = len(args.channels)
        elif args.rgb_only or args.uv_only:
            in_channels = 3
        else:
            in_channels = 6  # Default for RGB+UV
            print(f"Using {in_channels} channels")
        
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

    config["augmentations"]["strategy"] = args.augmentation_strategy

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
    
    if args.visualize:
        print("Generating augmentation visualization...")
        datamodule = SimCLRDataModule(config, rgb_only=args.rgb_only, uv_only=args.uv_only, usml=args.usml)
        visualize_augmentations(datamodule)
        print("Visualization saved in augmentation_examples/")
        return


    # Set up dataloader and model for training
    dm = SimCLRDataModule(
        config,
        rgb_only=args.rgb_only,
        uv_only=args.uv_only,
        usml=args.usml,
        channels=args.channels,
        # augmentation_strategy=args.augmentation_strategy,
    )
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
    
    trainer = pl.Trainer(**trainer_config, logger=logger)
    trainer.fit(model, datamodule=dm)

    # Log configuration
    with open(os.path.join(logger.log_dir, "config.json"), 'w') as f:
        json.dump(config, f)
    
    # Set up dataloader and model for inference then generate embeddings
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
    df.to_csv(
        os.path.join(logger.log_dir, output_filename),
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        float_format='%.15f')

    # Create sample plots
    plot_knn_examples(embeddings, filenames, data_dir)

    # Save plots to output dir
    plt.savefig(os.path.join(logger.log_dir, "knn_examples.png"))



def test_model_with_dummy_input(model, in_channels=6):
    """Test if the model works correctly with a dummy input."""
    print("\n===== Testing model with dummy input =====")
    # Create a dummy input tensor
    dummy_input = torch.randn(1, in_channels, 224, 224)
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    # Set model to eval mode
    model.eval()
    
    try:
        # Forward pass
        with torch.no_grad():
            # Test full model        output = model(dummy_input)
            print(f"Full model output shape: {output.shape}")
            print(f"Output contains NaN: {torch.isnan(output).any().item()}")
        
        print("Model test successful!")
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False



if __name__ == "__main__":
    args = get_args()
    main(args)