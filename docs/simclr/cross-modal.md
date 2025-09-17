# Cross-Modal Learning

Cross-modal contrastive learning enables comparison between different spectral modalities (RGB vs UV) within the same framework, providing insights into how different parts of the electromagnetic spectrum encode biological information.

## Overview

Traditional SimCLR contrasts different augmented views of the same image. Cross-modal learning extends this by contrasting RGB channels against UV channels, allowing the model to learn relationships between visible and ultraviolet patterns in biological organisms.

## Conceptual Framework

### Standard SimCLR
```
Image → [Augmentation 1, Augmentation 2] → Contrastive Loss
```

### Cross-Modal SimCLR
```
Multispectral Image → [RGB Channels, UV Channels] → Cross-Modal Contrastive Loss
```

## Implementation Approaches

### 1. Separate Modality Encoders

Use different encoders for RGB and UV channels:

```python
class CrossModalSimCLR(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.rgb_encoder = create_encoder(backbone, input_channels=3)
        self.uv_encoder = create_encoder(backbone, input_channels=3)
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, rgb_data, uv_data):
        rgb_features = self.rgb_encoder(rgb_data)
        uv_features = self.uv_encoder(uv_data)
        
        rgb_projected = self.projection_head(rgb_features)
        uv_projected = self.projection_head(uv_features)
        
        return rgb_projected, uv_projected
```

### 2. Shared Encoder with Modality Tokens

```python
class SharedModalityEncoder(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.encoder = create_encoder(backbone, input_channels=6)
        self.modality_tokens = nn.Parameter(torch.randn(2, 128))  # RGB, UV tokens
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
    
    def forward(self, multispectral_data):
        # Split into RGB and UV
        rgb_data = multispectral_data[:, :3]
        uv_data = multispectral_data[:, 3:6]
        
        # Process with shared encoder
        features = self.encoder(multispectral_data)
        
        # Apply modality-specific attention
        rgb_features = self.attention(features, self.modality_tokens[0:1])
        uv_features = self.attention(features, self.modality_tokens[1:2])
        
        return rgb_features, uv_features
```

## Training Configuration

### Basic Cross-Modal Config
```yaml
# config_cross_modal.yaml
data_dir: "data/multispectral"
out_dir: "outputs/cross_modal"

# Model configuration
backbone: "vit_l_base_patch16"
fusion_type: "separate"  # "concat", "separate", "attention"
use_modality_specific: true

# Cross-modal parameters
cross_modal_weight: 1.0
intra_modal_weight: 0.5
temperature: 0.1

# Training parameters
lr: 0.001
batch_size: 16
max_epochs: 100
```

### Fusion Strategy Options

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `concat` | Concatenate RGB and UV channels | Simple fusion |
| `separate` | Separate encoders for each modality | Maximum flexibility |
| `attention` | Attention-based fusion | Adaptive weighting |

## Training Commands

### Basic Cross-Modal Training
```bash
python scripts/simclr_birdcolour_kornia_spectral_multimodal.py \
  --config configs/config_cross_modal.yaml \
  --fusion-type separate
```

### Attention-Based Fusion
```bash
python scripts/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type attention \
  --use-modality-specific
```

### Weighted Loss Training
```bash
python scripts/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type separate \
  --cross-modal-weight 0.8 \
  --intra-modal-weight 0.2
```

## Loss Functions

### Cross-Modal Contrastive Loss
```python
def cross_modal_contrastive_loss(rgb_features, uv_features, temperature=0.1):
    """
    Contrastive loss between RGB and UV modalities.
    Positive pairs: RGB and UV from same image
    Negative pairs: RGB from one image, UV from different images
    """
    batch_size = rgb_features.shape[0]
    
    # Normalize features
    rgb_norm = F.normalize(rgb_features, dim=1)
    uv_norm = F.normalize(uv_features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(rgb_norm, uv_norm.T) / temperature
    
    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(batch_size).to(rgb_features.device)
    
    # Cross-entropy loss
    loss_rgb_to_uv = F.cross_entropy(similarity_matrix, labels)
    loss_uv_to_rgb = F.cross_entropy(similarity_matrix.T, labels)
    
    return (loss_rgb_to_uv + loss_uv_to_rgb) / 2
```

### Combined Loss Strategy
```python
def combined_cross_modal_loss(rgb_features, uv_features, 
                             cross_weight=0.7, intra_weight=0.3):
    """
    Combine cross-modal and intra-modal losses.
    """
    # Cross-modal loss (RGB vs UV)
    cross_modal_loss = cross_modal_contrastive_loss(rgb_features, uv_features)
    
    # Intra-modal losses (within RGB, within UV)
    rgb_intra_loss = standard_simclr_loss(rgb_features)
    uv_intra_loss = standard_simclr_loss(uv_features)
    intra_modal_loss = (rgb_intra_loss + uv_intra_loss) / 2
    
    # Combined loss
    total_loss = cross_weight * cross_modal_loss + intra_weight * intra_modal_loss
    
    return total_loss, cross_modal_loss, intra_modal_loss
```

## Advanced Techniques

### Modality-Specific Augmentations
```python
class CrossModalAugmentation:
    def __init__(self):
        # RGB-specific augmentations
        self.rgb_transforms = K.AugmentationSequential(
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)
        )
        
        # UV-specific augmentations
        self.uv_transforms = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0, std=0.1, p=0.3),
            K.RandomGamma((0.8, 1.2), p=0.5)
        )
        
        # Shared geometric augmentations
        self.shared_transforms = K.AugmentationSequential(
            K.RandomResizedCrop(224, scale=(0.2, 1.0)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(15)
        )
    
    def __call__(self, multispectral_data):
        # Apply shared geometric transforms
        transformed = self.shared_transforms(multispectral_data)
        
        # Split and apply modality-specific transforms
        rgb_data = transformed[:, :3]
        uv_data = transformed[:, 3:6]
        
        rgb_augmented = self.rgb_transforms(rgb_data)
        uv_augmented = self.uv_transforms(uv_data)
        
        return rgb_augmented, uv_augmented
```

### Adaptive Temperature Scaling
```python
class AdaptiveTemperature(nn.Module):
    def __init__(self, initial_temp=0.1):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(initial_temp)))
    
    def forward(self):
        return torch.exp(self.log_temperature)

# Usage in loss function
adaptive_temp = AdaptiveTemperature()
temperature = adaptive_temp()
loss = cross_modal_contrastive_loss(rgb_features, uv_features, temperature)
```

### Curriculum Learning
```python
class CrossModalCurriculum:
    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs
    
    def get_loss_weights(self, current_epoch):
        """Gradually increase cross-modal weight during training."""
        progress = current_epoch / self.total_epochs
        
        # Start with more intra-modal learning, gradually add cross-modal
        cross_weight = 0.2 + 0.6 * progress  # 0.2 → 0.8
        intra_weight = 0.8 - 0.6 * progress  # 0.8 → 0.2
        
        return cross_weight, intra_weight
```

## Evaluation Strategies

### Modality Alignment Analysis
```python
def evaluate_modality_alignment(rgb_embeddings, uv_embeddings, labels):
    """Evaluate how well RGB and UV embeddings align."""
    from scipy.stats import spearmanr
    
    # Compute pairwise similarities within each modality
    rgb_sim = cosine_similarity(rgb_embeddings)
    uv_sim = cosine_similarity(uv_embeddings)
    
    # Compute correlation between similarity matrices
    correlation, p_value = spearmanr(rgb_sim.flatten(), uv_sim.flatten())
    
    return {
        'modality_correlation': correlation,
        'p_value': p_value,
        'alignment_strength': 'strong' if correlation > 0.7 else 'moderate' if correlation > 0.4 else 'weak'
    }
```

### Cross-Modal Retrieval
```python
def cross_modal_retrieval_accuracy(rgb_embeddings, uv_embeddings, k=5):
    """Evaluate cross-modal retrieval performance."""
    n_samples = len(rgb_embeddings)
    correct_retrievals = 0
    
    for i in range(n_samples):
        # Use RGB embedding to retrieve UV embeddings
        rgb_query = rgb_embeddings[i:i+1]
        similarities = cosine_similarity(rgb_query, uv_embeddings)[0]
        
        # Get top-k most similar UV embeddings
        top_k_indices = np.argsort(similarities)[-k:]
        
        # Check if correct UV embedding (same index) is in top-k
        if i in top_k_indices:
            correct_retrievals += 1
    
    return correct_retrievals / n_samples
```

### Modality-Specific Classification
```python
def evaluate_modality_specific_performance(rgb_embeddings, uv_embeddings, labels):
    """Compare classification performance of each modality."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    # RGB classification
    rgb_classifier = LogisticRegression()
    rgb_scores = cross_val_score(rgb_classifier, rgb_embeddings, labels, cv=5)
    
    # UV classification
    uv_classifier = LogisticRegression()
    uv_scores = cross_val_score(uv_classifier, uv_embeddings, labels, cv=5)
    
    # Combined classification
    combined_embeddings = np.concatenate([rgb_embeddings, uv_embeddings], axis=1)
    combined_classifier = LogisticRegression()
    combined_scores = cross_val_score(combined_classifier, combined_embeddings, labels, cv=5)
    
    return {
        'rgb_accuracy': rgb_scores.mean(),
        'uv_accuracy': uv_scores.mean(),
        'combined_accuracy': combined_scores.mean(),
        'improvement': combined_scores.mean() - max(rgb_scores.mean(), uv_scores.mean())
    }
```

## Visualization Techniques

### Modality Comparison Plots
```python
def plot_modality_comparison(rgb_embeddings, uv_embeddings, labels):
    """Create side-by-side UMAP plots for RGB and UV embeddings."""
    from umap import UMAP
    
    # Reduce dimensions
    umap_reducer = UMAP(n_components=2, random_state=42)
    rgb_2d = umap_reducer.fit_transform(rgb_embeddings)
    uv_2d = umap_reducer.fit_transform(uv_embeddings)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RGB plot
    scatter1 = ax1.scatter(rgb_2d[:, 0], rgb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    ax1.set_title('RGB Embeddings')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    
    # UV plot
    scatter2 = ax2.scatter(uv_2d[:, 0], uv_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    ax2.set_title('UV Embeddings')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    plt.colorbar(scatter1, ax=ax1)
    plt.colorbar(scatter2, ax=ax2)
    plt.tight_layout()
    plt.savefig('modality_comparison.png', dpi=300)
```

### Cross-Modal Similarity Heatmap
```python
def plot_cross_modal_similarity(rgb_embeddings, uv_embeddings, labels):
    """Plot similarity matrix between RGB and UV embeddings."""
    # Compute cross-modal similarity matrix
    similarity_matrix = cosine_similarity(rgb_embeddings, uv_embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', cbar=True)
    plt.title('Cross-Modal Similarity Matrix (RGB vs UV)')
    plt.xlabel('UV Embeddings')
    plt.ylabel('RGB Embeddings')
    plt.savefig('cross_modal_similarity.png', dpi=300)
```

## Best Practices

### Training Strategy
1. Start with balanced loss weights (0.5 cross-modal, 0.5 intra-modal)
2. Use curriculum learning to gradually emphasize cross-modal learning
3. Monitor both modalities for balanced learning
4. Validate on cross-modal retrieval tasks

### Hyperparameter Tuning
- **Temperature**: Start with 0.1, adjust based on modality similarity
- **Loss weights**: Balance based on downstream task requirements
- **Learning rate**: May need different rates for different modalities
- **Batch size**: Ensure sufficient negative samples for contrastive learning

### Data Considerations
1. Ensure proper channel registration between RGB and UV
2. Verify that both modalities contain meaningful information
3. Check for modality-specific artifacts or noise
4. Balance dataset across species/categories

## Troubleshooting

### Common Issues

**One Modality Dominates Learning**
```python
# Monitor gradient magnitudes per modality
def monitor_gradient_balance(model):
    rgb_grad_norm = torch.norm(model.rgb_encoder.parameters().grad)
    uv_grad_norm = torch.norm(model.uv_encoder.parameters().grad)
    print(f"RGB grad norm: {rgb_grad_norm:.4f}, UV grad norm: {uv_grad_norm:.4f}")
```

**Poor Cross-Modal Alignment**
- Increase cross-modal loss weight
- Reduce temperature parameter
- Check data quality and registration
- Use stronger augmentations

**Memory Issues with Dual Encoders**
```yaml
# Use gradient checkpointing
gradient_checkpointing: true
# Reduce batch size
batch_size: 8
# Use mixed precision
precision: 16
```

## Example Workflows

### Comparative Study
```bash
# Train RGB-only baseline
python scripts/simclr_kornia_spectral.py --rgb-only

# Train UV-only baseline  
python scripts/simclr_kornia_spectral.py --uv-only

# Train cross-modal model
python scripts/simclr_birdcolour_kornia_spectral_multimodal.py \
  --fusion-type separate

# Compare results
python eval_vis/compare_cross_modal.py \
  --rgb-model rgb_model.ckpt \
  --uv-model uv_model.ckpt \
  --cross-modal-model cross_modal_model.ckpt
```

### Ablation Study
```bash
# Test different fusion strategies
for fusion in concat separate attention; do
    python scripts/simclr_birdcolour_kornia_spectral_multimodal.py \
      --fusion-type $fusion \
      --output-dir outputs/fusion_$fusion
done
```

## Next Steps

- **[Data Augmentation](augmentation.md)**: Modality-specific augmentation strategies
- **[Evaluation](../eval/overview.md)**: Cross-modal evaluation techniques
- **[Examples](../examples/example_multispectral.md)**: Practical cross-modal applications
