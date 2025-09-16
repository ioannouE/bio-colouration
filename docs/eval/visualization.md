# Visualization Tools

Comprehensive visualization tools for analyzing embeddings, model performance, and biological patterns in the Phenoscape framework. This guide covers static plots, interactive visualizations, and specialized tools for multispectral and hyperspectral data.

## Overview

Effective visualization is crucial for understanding model behavior, embedding quality, and biological relationships. Phenoscape provides tools for creating publication-ready figures and interactive exploratory visualizations.

## Core Visualization Functions

### Embedding Scatter Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def plot_embedding_scatter(embeddings, labels, method='pca', figsize=(12, 8)):
    """Create scatter plot of embeddings with dimensionality reduction."""
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = f'PCA Visualization (Explained Variance: {reducer.explained_variance_ratio_.sum():.2f})'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = 't-SNE Visualization'
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = 'UMAP Visualization'
    
    # Create plot
    plt.figure(figsize=figsize)
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return reduced_embeddings
```

### Embedding Heatmaps

```python
def plot_embedding_heatmap(embeddings, labels, sample_size=100):
    """Create heatmap visualization of embedding patterns."""
    
    # Sample embeddings if too large
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
    
    # Sort by labels for better visualization
    sorted_indices = np.argsort(labels)
    embeddings_sorted = embeddings[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(embeddings_sorted.T, 
                cmap='viridis', 
                cbar_kws={'label': 'Embedding Value'},
                xticklabels=False,
                yticklabels=False)
    
    # Add label annotations
    unique_labels = sorted(list(set(labels_sorted)))
    label_positions = []
    for label in unique_labels:
        positions = [i for i, l in enumerate(labels_sorted) if l == label]
        if positions:
            label_positions.append((np.mean(positions), label))
    
    for pos, label in label_positions:
        plt.axvline(x=pos, color='red', alpha=0.5, linestyle='--')
        plt.text(pos, -5, label, rotation=45, ha='center')
    
    plt.title('Embedding Heatmap by Label')
    plt.xlabel('Samples (sorted by label)')
    plt.ylabel('Embedding Dimensions')
    plt.tight_layout()
```

## Interactive Visualizations

### Plotly Interactive Scatter

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_interactive_scatter(embeddings_2d, labels, metadata=None, 
                             title="Interactive Embedding Visualization"):
    """Create interactive scatter plot with hover information."""
    
    # Prepare data
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Add metadata if provided
    if metadata is not None:
        for key, values in metadata.items():
            df[key] = values
    
    # Create interactive plot
    fig = px.scatter(df, x='x', y='y', color='label',
                    hover_data=df.columns.tolist(),
                    title=title,
                    width=800, height=600)
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        legend_title='Labels'
    )
    
    return fig

def create_3d_interactive_scatter(embeddings_3d, labels, metadata=None):
    """Create 3D interactive scatter plot."""
    
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'label': labels
    })
    
    if metadata is not None:
        for key, values in metadata.items():
            df[key] = values
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                       hover_data=df.columns.tolist(),
                       title='3D Interactive Embedding Visualization')
    
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    
    return fig
```

### Image Grid Visualization

```python
def plot_image_grid_with_embeddings(images, embeddings, labels, 
                                   grid_size=(10, 10), image_size=(64, 64)):
    """Plot grid of images arranged by embedding similarity."""
    
    # Reduce embeddings to 2D
    reducer = TSNE(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Normalize coordinates to grid
    x_coords = embeddings_2d[:, 0]
    y_coords = embeddings_2d[:, 1]
    
    x_norm = ((x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * (grid_size[0] - 1)).astype(int)
    y_norm = ((y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * (grid_size[1] - 1)).astype(int)
    
    # Create grid
    fig, axes = plt.subplots(grid_size[1], grid_size[0], 
                            figsize=(grid_size[0] * 2, grid_size[1] * 2))
    
    # Initialize grid with empty plots
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            axes[i, j].axis('off')
    
    # Place images in grid
    for idx, (x, y) in enumerate(zip(x_norm, y_norm)):
        if idx < len(images):
            img = images[idx]
            if len(img.shape) == 3 and img.shape[0] <= 3:  # Channel first
                img = img.transpose(1, 2, 0)
            
            axes[y, x].imshow(img)
            axes[y, x].set_title(f'{labels[idx]}', fontsize=8)
            axes[y, x].axis('off')
    
    plt.suptitle('Image Grid Arranged by Embedding Similarity', fontsize=16)
    plt.tight_layout()
```

## Specialized Visualizations

### Cross-Modal Comparison

```python
def plot_cross_modal_comparison(rgb_embeddings, uv_embeddings, labels):
    """Visualize RGB vs UV embedding relationships."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PCA comparison
    pca_rgb = PCA(n_components=2).fit_transform(rgb_embeddings)
    pca_uv = PCA(n_components=2).fit_transform(uv_embeddings)
    
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # RGB PCA
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        axes[0, 0].scatter(pca_rgb[mask, 0], pca_rgb[mask, 1], 
                          c=[colors[i]], label=label, alpha=0.7)
    axes[0, 0].set_title('RGB Embeddings (PCA)')
    axes[0, 0].legend()
    
    # UV PCA
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        axes[0, 1].scatter(pca_uv[mask, 0], pca_uv[mask, 1], 
                          c=[colors[i]], label=label, alpha=0.7)
    axes[0, 1].set_title('UV Embeddings (PCA)')
    
    # Cross-modal correlation
    correlations = []
    for i in range(rgb_embeddings.shape[1]):
        corr = np.corrcoef(rgb_embeddings[:, i], uv_embeddings[:, i])[0, 1]
        correlations.append(corr)
    
    axes[0, 2].plot(correlations)
    axes[0, 2].set_title('Dimension-wise RGB-UV Correlations')
    axes[0, 2].set_xlabel('Embedding Dimension')
    axes[0, 2].set_ylabel('Correlation')
    
    # Similarity distribution
    similarities = []
    for i in range(len(rgb_embeddings)):
        sim = np.dot(rgb_embeddings[i], uv_embeddings[i]) / (
            np.linalg.norm(rgb_embeddings[i]) * np.linalg.norm(uv_embeddings[i])
        )
        similarities.append(sim)
    
    axes[1, 0].hist(similarities, bins=50, alpha=0.7)
    axes[1, 0].set_title('RGB-UV Cosine Similarity Distribution')
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_ylabel('Frequency')
    
    # Scatter plot of first dimensions
    axes[1, 1].scatter(rgb_embeddings[:, 0], uv_embeddings[:, 0], alpha=0.5)
    axes[1, 1].set_xlabel('RGB Embedding (Dim 0)')
    axes[1, 1].set_ylabel('UV Embedding (Dim 0)')
    axes[1, 1].set_title('RGB vs UV (First Dimension)')
    
    # Mean embedding comparison
    rgb_means = [rgb_embeddings[np.array(labels) == label].mean(axis=0) 
                 for label in unique_labels]
    uv_means = [uv_embeddings[np.array(labels) == label].mean(axis=0) 
                for label in unique_labels]
    
    mean_similarities = []
    for rgb_mean, uv_mean in zip(rgb_means, uv_means):
        sim = np.dot(rgb_mean, uv_mean) / (np.linalg.norm(rgb_mean) * np.linalg.norm(uv_mean))
        mean_similarities.append(sim)
    
    axes[1, 2].bar(range(len(unique_labels)), mean_similarities)
    axes[1, 2].set_xticks(range(len(unique_labels)))
    axes[1, 2].set_xticklabels(unique_labels, rotation=45)
    axes[1, 2].set_title('Mean RGB-UV Similarity by Label')
    axes[1, 2].set_ylabel('Cosine Similarity')
    
    plt.tight_layout()
```

### Spectral Band Visualization

```python
def plot_spectral_bands(hyperspectral_data, wavelengths, sample_indices=None):
    """Visualize hyperspectral data across wavelengths."""
    
    if sample_indices is None:
        sample_indices = np.random.choice(len(hyperspectral_data), 5, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Spectral signatures
    for idx in sample_indices:
        spectrum = hyperspectral_data[idx].mean(axis=(1, 2))  # Average over spatial dimensions
        axes[0, 0].plot(wavelengths, spectrum, alpha=0.7, label=f'Sample {idx}')
    
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Reflectance')
    axes[0, 0].set_title('Spectral Signatures')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean spectrum across all samples
    mean_spectrum = hyperspectral_data.mean(axis=(0, 2, 3))
    std_spectrum = hyperspectral_data.std(axis=(0, 2, 3))
    
    axes[0, 1].plot(wavelengths, mean_spectrum, 'b-', label='Mean')
    axes[0, 1].fill_between(wavelengths, 
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           alpha=0.3, label='±1 STD')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Reflectance')
    axes[0, 1].set_title('Mean Spectral Response')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Band variance
    band_variance = hyperspectral_data.var(axis=(0, 2, 3))
    axes[1, 0].plot(wavelengths, band_variance)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title('Spectral Band Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # RGB composite
    # Find closest wavelengths to RGB
    rgb_wavelengths = [650, 550, 450]  # Approximate R, G, B
    rgb_indices = [np.argmin(np.abs(wavelengths - w)) for w in rgb_wavelengths]
    
    rgb_composite = np.stack([
        hyperspectral_data[sample_indices[0], rgb_indices[0]],
        hyperspectral_data[sample_indices[0], rgb_indices[1]],
        hyperspectral_data[sample_indices[0], rgb_indices[2]]
    ], axis=-1)
    
    # Normalize for display
    rgb_composite = (rgb_composite - rgb_composite.min()) / (rgb_composite.max() - rgb_composite.min())
    
    axes[1, 1].imshow(rgb_composite)
    axes[1, 1].set_title('RGB Composite (Sample 0)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
```

## Training Visualization

### Loss Curves

```python
def plot_training_curves(train_losses, val_losses=None, learning_rates=None):
    """Plot training and validation loss curves."""
    
    fig, axes = plt.subplots(1, 2 if learning_rates is not None else 1, 
                            figsize=(15 if learning_rates is not None else 10, 5))
    
    if learning_rates is None:
        axes = [axes]
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate schedule
    if learning_rates is not None:
        axes[1].plot(epochs, learning_rates, 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
```

### Augmentation Preview

```python
def plot_augmentation_examples(original_images, augmented_images, n_examples=4):
    """Show examples of data augmentations."""
    
    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 4, 8))
    
    for i in range(n_examples):
        # Original image
        orig_img = original_images[i]
        if len(orig_img.shape) == 3 and orig_img.shape[0] <= 3:
            orig_img = orig_img.transpose(1, 2, 0)
        
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Augmented image
        aug_img = augmented_images[i]
        if len(aug_img.shape) == 3 and aug_img.shape[0] <= 3:
            aug_img = aug_img.transpose(1, 2, 0)
        
        axes[1, i].imshow(aug_img)
        axes[1, i].set_title(f'Augmented {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16)
    plt.tight_layout()
```

## Statistical Visualizations

### Correlation Matrices

```python
def plot_correlation_matrix(embeddings, labels, method='pearson'):
    """Plot correlation matrix of embedding dimensions."""
    
    if method == 'pearson':
        corr_matrix = np.corrcoef(embeddings.T)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(embeddings, axis=0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=False, 
                cmap='coolwarm', 
                center=0,
                square=True,
                cbar_kws={'label': f'{method.capitalize()} Correlation'})
    
    plt.title(f'Embedding Dimension {method.capitalize()} Correlations')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Dimension')
    plt.tight_layout()
```

### Distribution Plots

```python
def plot_embedding_distributions(embeddings, labels, n_dims=6):
    """Plot distributions of embedding dimensions by label."""
    
    n_dims = min(n_dims, embeddings.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for dim in range(n_dims):
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            dim_values = embeddings[mask, dim]
            
            axes[dim].hist(dim_values, bins=30, alpha=0.6, 
                          color=colors[i], label=label, density=True)
        
        axes[dim].set_title(f'Dimension {dim} Distribution')
        axes[dim].set_xlabel('Embedding Value')
        axes[dim].set_ylabel('Density')
        axes[dim].legend()
        axes[dim].grid(True, alpha=0.3)
    
    plt.tight_layout()
```

## Export and Saving

### High-Quality Figure Export

```python
def save_publication_figure(fig, filename, dpi=300, formats=['png', 'pdf']):
    """Save figure in publication-ready formats."""
    
    for fmt in formats:
        filepath = f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")

def create_figure_grid(plot_functions, data, grid_shape=(2, 2), figsize=(16, 12)):
    """Create a grid of multiple plots."""
    
    fig = plt.figure(figsize=figsize)
    
    for i, plot_func in enumerate(plot_functions):
        plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        plot_func(data)
    
    plt.tight_layout()
    return fig
```

## Visualization Pipeline

### Complete Visualization Workflow

```python
def create_comprehensive_visualization_report(embeddings, labels, metadata=None, 
                                            output_dir='visualization_report'):
    """Generate comprehensive visualization report."""
    
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Embedding scatter plots
    print("Creating embedding visualizations...")
    for method in ['pca', 'tsne', 'umap']:
        plt.figure(figsize=(12, 8))
        reduced_emb = plot_embedding_scatter(embeddings, labels, method=method)
        plt.savefig(output_dir / f'embedding_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Correlation matrix
    print("Creating correlation matrix...")
    plot_correlation_matrix(embeddings, labels)
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution plots
    print("Creating distribution plots...")
    plot_embedding_distributions(embeddings, labels)
    plt.savefig(output_dir / 'embedding_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Interactive plots
    print("Creating interactive visualizations...")
    pca_2d = PCA(n_components=2).fit_transform(embeddings)
    interactive_fig = create_interactive_scatter(pca_2d, labels, metadata)
    interactive_fig.write_html(output_dir / 'interactive_embeddings.html')
    
    # 5. 3D interactive plot
    pca_3d = PCA(n_components=3).fit_transform(embeddings)
    interactive_3d_fig = create_3d_interactive_scatter(pca_3d, labels, metadata)
    interactive_3d_fig.write_html(output_dir / 'interactive_embeddings_3d.html')
    
    print(f"Visualization report saved to {output_dir}")
    
    return {
        'output_directory': output_dir,
        'files_created': list(output_dir.glob('*')),
        'summary': {
            'n_samples': len(embeddings),
            'n_dimensions': embeddings.shape[1],
            'n_labels': len(set(labels))
        }
    }
```

## Next Steps

- **[Statistical Metrics](statistical-metrics.md)**: Quantitative analysis methods
- **[Examples](../examples/)**: Practical visualization examples
- **[Embedding Analysis](embedding-analysis.md)**: Advanced embedding analysis techniques
