import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import datetime as dt
import json
import os
import sys
# Add the parent directory of 'train' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
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

from pytorch_lightning.loggers import TensorBoardLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Import Kornia augmentations
import kornia.augmentation as K
from kornia.constants import BorderType
from kornia.geometry.transform import get_perspective_transform, warp_perspective
import kornia 

from data.multispectral_data import SimCLRDataModule
import pytz
import torch.nn.functional as F

import tempfile
# Set temporary directory on same filesystem as checkpoint destination
# tmpdir = "/shared/cooney_lab/Shared/Eleftherios-Ioannou/tmp"
# os.environ["TMPDIR"] = tmpdir
# os.environ["FSSPEC_TEMP"] = tmpdir
# tempfile.tempdir = tmpdir



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

class ModalityFusion(nn.Module):
    """Fusion mechanism for combining features from different modalities."""
    
    def __init__(self, fusion_type: str, feature_dim: int):
        """
        Initialize fusion module
        
        Args:
            fusion_type: Type of fusion - 'concat', 'separate', 'attention'
            feature_dim: Dimension of input features
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim
        
        if fusion_type == 'attention':
            # Create cross-attention mechanism
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
            self.scale = feature_dim ** -0.5
            self.attn_dropout = nn.Dropout(0.1)
            self.output_proj = nn.Linear(feature_dim, feature_dim)
        elif fusion_type == 'concat':
            # Create projection for concatenated features
            self.concat_proj = nn.Linear(feature_dim * 2, feature_dim)
            
    def forward(self, rgb_features, uv_features):
        """
        Fuse RGB and UV features based on fusion type
        
        Args:
            rgb_features: Features from RGB modality [batch_size, feature_dim]
            uv_features: Features from UV modality [batch_size, feature_dim]
            
        Returns:
            Fused features or separate features depending on fusion type
        """
        if self.fusion_type == 'separate':
            # Return features separately without fusion
            return rgb_features, uv_features
            
        elif self.fusion_type == 'concat':
            # Concatenate features and project to original dimension
            concat_features = torch.cat([rgb_features, uv_features], dim=1)
            fused_features = self.concat_proj(concat_features)
            return fused_features
            
        elif self.fusion_type == 'attention':
            # Cross-attention mechanism
            batch_size = rgb_features.shape[0]
            
            # Project to query, key, value
            q = self.query_proj(rgb_features).view(batch_size, 1, self.feature_dim)  # RGB as query
            k = self.key_proj(uv_features).view(batch_size, 1, self.feature_dim)     # UV as key
            v = self.value_proj(uv_features).view(batch_size, 1, self.feature_dim)   # UV as value
            
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_probs = self.attn_dropout(F.softmax(attn_scores, dim=-1))
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v).view(batch_size, self.feature_dim)
            
            # Combine with original RGB features (residual connection)
            fused_features = rgb_features + self.output_proj(attn_output)
            return fused_features
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

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
        
        # Add cross-modal loss weight
        self.cross_modal_weight = criterion_config.get('cross_modal_weight', 1.0)
        
        # Get fusion configuration
        self.use_modality_specific = config.get('use_modality_specific', False)
        self.fusion_type = config.get('fusion_type', 'separate')  # 'concat', 'separate', 'attention'
        
        print(f"Using loss={self.loss_type} with weights: Global NTXent={self.global_weight}, Local NTXent={self.local_weight}, Cross-Modal NTXent={self.cross_modal_weight}")
        print(f"Using modality-specific encoders: {self.use_modality_specific}")
        if self.use_modality_specific:
            print(f"Fusion type: {self.fusion_type}")

        # Flag to determine if we should use local patch-level loss
        self.use_local_loss = criterion_config.get('use_local_loss', False)  
        print(f"Using local loss: {self.use_local_loss}")
        
        # Flag to determine if we should use cross-modal loss
        self.use_cross_modal = self.loss_type == 'crossmodal'
        print(f"Using cross-modal loss: {self.use_cross_modal}")
        
        # create a backbone and remove the classification head
        model = config.get("backbone")
        weights = config.get('weights')

        if model == 'resnet50':
            base_model = torchvision.models.get_model(model, weights=weights)
            backbone = nn.Sequential(*list(base_model.children())[:-1])
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
        
        # Setup modality-specific or shared encoders
        if self.use_modality_specific:
            # Setup separate encoders for RGB and UV modalities
            if model == 'resnet50':
                # For ResNet, create two separate backbones
                rgb_base_model = torchvision.models.get_model(model, weights=weights)
                uv_base_model = torchvision.models.get_model(model, weights=weights)
                self.rgb_backbone = nn.Sequential(*list(rgb_base_model.children())[:-1])
                self.uv_backbone = nn.Sequential(*list(uv_base_model.children())[:-1])
            else:
                # For ViT, create two separate models
                rgb_base_model = vit_l_16(weights=weights, image_size=224)
                uv_base_model = vit_l_16(weights=weights, image_size=224)
                
                # Modify input channels for UV model (still use 3 channels, but different initialization)
                rgb_base_model.heads = nn.Identity()
                uv_base_model.heads = nn.Identity()
                
                self.rgb_backbone = rgb_base_model
                self.uv_backbone = uv_base_model
                
            # Create fusion mechanism
            self.fusion = ModalityFusion(self.fusion_type, hidden_dim)
            
            # Set backbone to be a reference to rgb_backbone for compatibility
            # This avoids creating a third unnecessary model
            self.backbone = self.rgb_backbone
        else:
            # Use single encoder for combined RGB+UV input
            if in_channels > 3:
                checkpoint_path = config.get('checkpoint_path', None)
                if model == 'resnet50':
                    self.backbone = MultispectralBackbone(base_model, in_channels=in_channels, checkpoint_path=checkpoint_path)
                else:
                    base_model = modify_vit_input_channels(base_model, in_channels=in_channels)
                    self.backbone = base_model
            else:
                self.backbone = base_model
        
        # Define the projection head
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        
        # Add separate projection heads for RGB and UV modalities
        if self.use_cross_modal:
            self.rgb_projection = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
            self.uv_projection = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        
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
        
        # Extract patch tokens if using ViT and local loss is enabled
        patch_tokens = None
        
        if self.use_modality_specific:
            # Split input into RGB and UV modalities
            rgb, uv = self.split_rgb_uv(x)
            
            # Process through separate encoders
            rgb_h = self.rgb_backbone(rgb)
            uv_h = self.uv_backbone(uv)
            
            # Ensure features are contiguous
            rgb_h = rgb_h.contiguous()
            uv_h = uv_h.contiguous()
            
            # Apply fusion based on strategy
            if self.fusion_type == 'separate':
                # Keep features separate for projection
                h = (rgb_h + uv_h) / 2  # Average for compatibility with non-modality specific code
                
                # Project using separate heads for cross-modal loss
                z = self.projection_head(h)
                rgb_z = self.rgb_projection(rgb_h) if self.use_cross_modal else None
                uv_z = self.uv_projection(uv_h) if self.use_cross_modal else None
                
            else:  # 'concat' or 'attention'
                # Fuse features using the specified mechanism
                h = self.fusion(rgb_h, uv_h)
                
                # Project fused features
                z = self.projection_head(h)
                
                # Also create modality-specific projections if needed
                rgb_z = self.rgb_projection(rgb_h) if self.use_cross_modal else None
                uv_z = self.uv_projection(uv_h) if self.use_cross_modal else None
        else:
            # Original behavior for non-modality-specific model
            if self.spectral_attn is not None:
                x = self.spectral_attn(x)
            
            if self.is_vit and self.use_local_loss:
                patch_tokens = self._extract_patch_tokens(x)
            
            # Get embeddings from backbone
            h = self.backbone(x)  # Output shape: [batch_size, hidden_dim]
            
            # Ensure h is contiguous before projection
            h = h.contiguous()
            
            # Project embeddings
            z = self.projection_head(h)
            
            # Create separate RGB and UV embeddings if using cross-modal loss
            rgb_z, uv_z = None, None
            if self.use_cross_modal:
                rgb_z = self.rgb_projection(h)
                uv_z = self.uv_projection(h)
        
        # If we have patch tokens, project them too
        local_z = None
        if patch_tokens is not None:
            # Project each patch token
            # patch_tokens shape: [batch_size, num_patches, hidden_dim]
            local_z = self.local_projection_head(patch_tokens)
            
        return z, local_z, rgb_z, uv_z
        
    def split_rgb_uv(self, x):
        """
        Split 6-channel input into RGB and UV modalities
        
        Args:
            x: Input tensor with shape [batch_size, 6, height, width]
            
        Returns:
            rgb: RGB channels [batch_size, 3, height, width]
            uv: UV channels [batch_size, 3, height, width]
        """
        # Check input channels
        if x.shape[1] != 6:
            raise ValueError(f"Expected 6-channel input for RGB-UV splitting, got {x.shape[1]} channels")
        
        # Split channels
        rgb = x[:, :3, :, :]  # First 3 channels are RGB
        uv = x[:, 3:6, :, :]  # Next 3 channels are UV
        
        return rgb, uv
    
    def compute_cross_modal_loss(self, rgb_z0, uv_z0, rgb_z1, uv_z1):
        """
        Compute cross-modal contrastive loss between RGB and UV representations
        
        Args:
            rgb_z0: RGB features from first view [batch_size, proj_dim]
            uv_z0: UV features from first view [batch_size, proj_dim]
            rgb_z1: RGB features from second view [batch_size, proj_dim]
            uv_z1: UV features from second view [batch_size, proj_dim]
            
        Returns:
            Cross-modal NTXent loss
        """
        # Compute cross-modal loss between RGB and UV from same view
        loss_rgb_uv_0 = self.ntxent_criterion(rgb_z0, uv_z0)
        loss_rgb_uv_1 = self.ntxent_criterion(rgb_z1, uv_z1)
        
        # Compute cross-modal loss between RGB and UV from different views
        loss_rgb0_uv1 = self.ntxent_criterion(rgb_z0, uv_z1)
        loss_rgb1_uv0 = self.ntxent_criterion(rgb_z1, uv_z0)
        
        # Average the cross-modal losses
        cross_modal_loss = (loss_rgb_uv_0 + loss_rgb_uv_1 + loss_rgb0_uv1 + loss_rgb1_uv0) / 4.0
        
        return cross_modal_loss
        
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
        
        # Forward pass through the model with both views
        z0, local_z0, rgb_z0, uv_z0 = self.forward(x0)
        z1, local_z1, rgb_z1, uv_z1 = self.forward(x1)
        
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
        
        # Compute cross-modal loss if enabled and weight > 0
        if self.cross_modal_weight > 0 and self.use_cross_modal and rgb_z0 is not None and uv_z0 is not None:
            cross_modal_loss = self.compute_cross_modal_loss(rgb_z0, uv_z0, rgb_z1, uv_z1)
            self.log("train_cross_modal_loss", cross_modal_loss)
            total_loss += self.cross_modal_weight * cross_modal_loss
            
        self.log("train_loss_ssl", total_loss)
        
        # Also log using wandb directly
        if wandb.run is not None:
            # Create a log dictionary with all metrics, including current epoch
            log_dict = {
                'train_loss_ssl': total_loss,
                'train_loss_global': global_loss,
                'train_loss_local': local_loss if self.use_local_loss else 0,
                'train_loss_cross_modal': cross_modal_loss if self.use_cross_modal else 0,
                'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                'epoch': self.current_epoch + 1  # Add current epoch (1-indexed)
            }
            
            # Log all metrics
            wandb.log(log_dict)
        
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
        
    def on_train_epoch_end(self):
        # Get current epoch number (0-based, so add 1 for reporting)
        current_epoch = self.current_epoch + 1
        
        # Get the average loss for this epoch from the metrics
        # Lightning automatically calculates epoch-level metrics from step-level metrics
        avg_loss = self.trainer.callback_metrics.get('train_loss_ssl', None)
        
        # Log epoch-level metrics to wandb
        if wandb.run is not None:
            # Create a metrics dict with explicit step=None to ensure this logs at epoch level
            # rather than incrementing the step counter
            metrics = {
                'epoch': current_epoch,
                'epoch_avg_loss': avg_loss
            }
            
            # If we have other loss components, log their epoch averages too
            global_loss = self.trainer.callback_metrics.get('train_global_loss', None)
            if global_loss is not None:
                metrics['epoch_avg_global_loss'] = global_loss
                
            if self.use_local_loss:
                local_loss = self.trainer.callback_metrics.get('train_local_loss', None)
                if local_loss is not None:
                    metrics['epoch_avg_local_loss'] = local_loss
                    
            if self.use_cross_modal:
                cross_modal_loss = self.trainer.callback_metrics.get('train_cross_modal_loss', None)
                if cross_modal_loss is not None:
                    metrics['epoch_avg_cross_modal_loss'] = cross_modal_loss
            
            # Log the epoch-level metrics, with commit=True to ensure it's logged immediately
            wandb.log(metrics, commit=True)
            
            # You can also use wandb.summary to track the best metrics across all epochs
            # This gets updated automatically when a better value is found
            if avg_loss is not None and (wandb.run.summary.get('best_epoch_loss', float('inf')) > avg_loss):
                wandb.run.summary['best_epoch_loss'] = avg_loss
                wandb.run.summary['best_epoch'] = current_epoch


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
            
            # Get embeddings using the appropriate backbone
            if model.use_modality_specific:
                # Split input and use RGB backbone
                rgb, _ = model.split_rgb_uv(img)
                embedding = model.rgb_backbone(rgb)
            else:
                # Use regular backbone
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
        choices=["global", "local", "combined", "crossmodal"],
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
        "--cross-modal-weight",
        type=float,
        default=1.0,
        help="Weight for cross-modal NTXentLoss",
    )
    parser.add_argument(
        "--use-modality-specific",
        action='store_true', 
        help='use separate encoders for RGB and UV modalities'
    )
    parser.add_argument(
        "--fusion-type",
        type=str, 
        default='separate', 
        choices=['concat', 'separate', 'attention'], 
        help='fusion type for modality-specific encoders'
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
    elif args.usml:
        print("USML mode enabled: Using tetrahedral colour space (5 channels)")
        config["data"]["usml"] = True
        in_channels = 5
    else:
        print("Using all channels (RGB+UV)")
        in_channels = 6
        
    # Add model checkpoint path to config if specified
    if args.model_checkpoint:
        print(f"Using pretrained MultispectralClassifier weights from: {args.model_checkpoint}")
        config["model"]["checkpoint_path"] = args.model_checkpoint

    # Set loss configuration in config
    if 'criterion' not in config["model"]:
        config["model"]['criterion'] = {}
    config["model"]['criterion']['type'] = args.loss_type
    
    # Add weights for combined loss
    if args.loss_type == 'combined':
        print(f"Using combined loss with weights: Global NTXent={args.global_weight}, Local NTXent={args.local_weight}")
        config["model"]['criterion']['global_weight'] = args.global_weight
        config["model"]['criterion']['local_weight'] = args.local_weight
    elif args.loss_type == 'crossmodal':
        if args.rgb_only:
            print("Warning: Cross-modal loss cannot be used in RGB-only mode. Falling back to global loss.")
            config["model"]['criterion']['type'] = 'global'  # Fallback to global loss
        else:
            print(f"Using cross-modal loss with weight: {args.cross_modal_weight} and global weight: {args.global_weight}")
            config["model"]['criterion']['cross_modal_weight'] = args.cross_modal_weight
            config["model"]['criterion']['global_weight'] = args.global_weight  # Keep some global loss
    else:
        print(f"Using {args.loss_type} loss")
        
    # Add modality-specific encoder settings to config
    config["model"]["use_modality_specific"] = args.use_modality_specific
    config["model"]["fusion_type"] = args.fusion_type
    if args.use_modality_specific:
        print(f"Using modality-specific encoders with fusion type: {args.fusion_type}")

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
    
    # Format a more readable date-time for the wandb run name
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize wandb directly
    os.environ["WANDB_CONSOLE"] = "off"
    wandb.init(
        name=f"simclr_{config['model'].get('backbone', 'vit')}_{timestamp}",
        project="simclr",
        dir=os.path.join(config["output"]["out_dir"], "wandb"),
    )
    
    # Add callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=os.path.join(config["output"]["out_dir"], "checkpoints", version),
            filename='{epoch:02d}-{train_loss_ssl:.4f}',
            save_top_k=3,
            monitor='train_loss_ssl',
            mode='min'
        )
    ]
    
    if args.visualize:
        print("Generating augmentation visualization...")
        datamodule = SimCLRDataModule(config)
        visualize_augmentations(datamodule)
        print("Visualization saved in augmentation_examples/")
        return


    # Set up dataloader and model for training
    dm = SimCLRDataModule(config, rgb_only=args.rgb_only, usml=args.usml)
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
    
    # Update trainer to use TensorBoard logger and callbacks
    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)

    # Save config to wandb
    wandb.config.update(config)
    
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
    
    # Log KNN examples to WandB
    wandb.log({"knn_examples": wandb.Image(os.path.join(logger.log_dir, "knn_examples.png"))})
    wandb.finish()


def test_model_with_dummy_input(model, in_channels=6):
    """Test if the model works correctly with a dummy input."""
    print("\n===== Testing model with dummy input =====")
    # Create a dummy input tensor
    dummy_input = torch.randn(1, in_channels, 224, 224)
    
    # Forward pass through the model (just the backbone first)
    with torch.no_grad():
        try:
            embedding = model.backbone(dummy_input)
            print(f"✓ Backbone output shape: {embedding.shape}")
            
            # Full forward pass
            z, local_z, rgb_z, uv_z = model(dummy_input)
            print(f"✓ Projection head output shape: {z.shape}")
            
            if local_z is not None:
                print(f"✓ Local projection output shape: {local_z.shape}")
                
            if rgb_z is not None:
                print(f"✓ RGB projection output shape: {rgb_z.shape}")
                
            if uv_z is not None:
                print(f"✓ UV projection output shape: {uv_z.shape}")
                
            print("Model test successful!")
            
        except Exception as e:
            print(f"Model test failed with error: {e}")
    print("=======================================")


if __name__ == "__main__":
    args = get_args()
    main(args)
