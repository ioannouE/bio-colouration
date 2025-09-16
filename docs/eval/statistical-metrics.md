# Statistical Metrics

Comprehensive statistical analysis methods for evaluating embedding quality, biological correlations, and model performance in the Phenoscape framework. This guide covers clustering metrics, correlation analysis, and hypothesis testing.

## Overview

Statistical metrics provide quantitative measures of embedding quality, biological relevance, and model performance. These metrics help validate that learned representations capture meaningful biological patterns and relationships.

## Clustering Metrics

### Silhouette Analysis

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt

def compute_silhouette_analysis(embeddings, labels):
    """Compute comprehensive silhouette analysis."""
    
    # Overall silhouette score
    silhouette_avg = silhouette_score(embeddings, labels)
    
    # Per-sample silhouette scores
    sample_silhouette_values = silhouette_samples(embeddings, labels)
    
    # Per-label statistics
    unique_labels = sorted(list(set(labels)))
    label_stats = {}
    
    for label in unique_labels:
        label_mask = np.array(labels) == label
        label_silhouettes = sample_silhouette_values[label_mask]
        
        label_stats[label] = {
            'mean_silhouette': np.mean(label_silhouettes),
            'std_silhouette': np.std(label_silhouettes),
            'min_silhouette': np.min(label_silhouettes),
            'max_silhouette': np.max(label_silhouettes),
            'n_samples': len(label_silhouettes)
        }
    
    return {
        'overall_silhouette': silhouette_avg,
        'sample_silhouettes': sample_silhouette_values,
        'label_statistics': label_stats
    }

def plot_silhouette_analysis(embeddings, labels, figsize=(12, 8)):
    """Create silhouette analysis plot."""
    
    silhouette_avg = silhouette_score(embeddings, labels)
    sample_silhouette_values = silhouette_samples(embeddings, labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Silhouette plot
    y_lower = 10
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
    
    for i, (label, color) in enumerate(zip(unique_labels, colors)):
        label_mask = np.array(labels) == label
        label_silhouettes = sample_silhouette_values[label_mask]
        label_silhouettes.sort()
        
        size_cluster_i = label_silhouettes.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, label_silhouettes,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10
    
    ax1.set_xlabel('Silhouette Coefficient Values')
    ax1.set_ylabel('Cluster Label')
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f'Average Score: {silhouette_avg:.3f}')
    ax1.legend()
    
    # Distribution of silhouette scores
    ax2.hist(sample_silhouette_values, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=silhouette_avg, color="red", linestyle="--", 
                label=f'Average: {silhouette_avg:.3f}')
    ax2.set_xlabel('Silhouette Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Silhouette Scores')
    ax2.legend()
    
    plt.tight_layout()
    return silhouette_avg
```

### Calinski-Harabasz Index

```python
from sklearn.metrics import calinski_harabasz_score

def compute_calinski_harabasz_score(embeddings, labels):
    """Compute Calinski-Harabasz index (variance ratio criterion)."""
    
    ch_score = calinski_harabasz_score(embeddings, labels)
    
    # Additional statistics
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    n_samples = len(embeddings)
    
    return {
        'calinski_harabasz_score': ch_score,
        'n_clusters': n_clusters,
        'n_samples': n_samples,
        'score_per_cluster': ch_score / n_clusters if n_clusters > 0 else 0
    }
```

### Davies-Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score

def compute_davies_bouldin_score(embeddings, labels):
    """Compute Davies-Bouldin index (lower is better)."""
    
    db_score = davies_bouldin_score(embeddings, labels)
    
    return {
        'davies_bouldin_score': db_score,
        'interpretation': 'lower_is_better'
    }
```

## Correlation Analysis

### Embedding-Metadata Correlations

```python
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

def compute_embedding_metadata_correlations(embeddings, metadata_df, methods=['pearson', 'spearman']):
    """Compute correlations between embeddings and metadata features."""
    
    results = {}
    
    for method in methods:
        method_results = {}
        
        for feature in metadata_df.columns:
            if metadata_df[feature].dtype in ['int64', 'float64']:
                # Numerical feature
                feature_correlations = []
                
                for dim in range(embeddings.shape[1]):
                    if method == 'pearson':
                        corr, p_val = pearsonr(embeddings[:, dim], metadata_df[feature])
                    elif method == 'spearman':
                        corr, p_val = spearmanr(embeddings[:, dim], metadata_df[feature])
                    elif method == 'kendall':
                        corr, p_val = kendalltau(embeddings[:, dim], metadata_df[feature])
                    
                    feature_correlations.append({
                        'dimension': dim,
                        'correlation': corr,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    })
                
                method_results[feature] = feature_correlations
        
        results[method] = method_results
    
    return results

def find_significant_correlations(correlation_results, significance_threshold=0.05, 
                                correlation_threshold=0.3):
    """Find significant correlations above threshold."""
    
    significant_correlations = []
    
    for method, method_results in correlation_results.items():
        for feature, correlations in method_results.items():
            for corr_data in correlations:
                if (corr_data['p_value'] < significance_threshold and 
                    abs(corr_data['correlation']) > correlation_threshold):
                    
                    significant_correlations.append({
                        'method': method,
                        'feature': feature,
                        'dimension': corr_data['dimension'],
                        'correlation': corr_data['correlation'],
                        'p_value': corr_data['p_value']
                    })
    
    return sorted(significant_correlations, 
                 key=lambda x: abs(x['correlation']), reverse=True)
```

### Cross-Modal Correlations

```python
def compute_cross_modal_correlations(rgb_embeddings, uv_embeddings):
    """Compute correlations between RGB and UV embeddings."""
    
    assert len(rgb_embeddings) == len(uv_embeddings), "Embedding arrays must have same length"
    
    # Overall correlation
    rgb_flat = rgb_embeddings.flatten()
    uv_flat = uv_embeddings.flatten()
    overall_corr, overall_p = pearsonr(rgb_flat, uv_flat)
    
    # Dimension-wise correlations
    dim_correlations = []
    for dim in range(rgb_embeddings.shape[1]):
        corr, p_val = pearsonr(rgb_embeddings[:, dim], uv_embeddings[:, dim])
        dim_correlations.append({
            'dimension': dim,
            'correlation': corr,
            'p_value': p_val
        })
    
    # Sample-wise similarities (cosine similarity)
    sample_similarities = []
    for i in range(len(rgb_embeddings)):
        similarity = np.dot(rgb_embeddings[i], uv_embeddings[i]) / (
            np.linalg.norm(rgb_embeddings[i]) * np.linalg.norm(uv_embeddings[i])
        )
        sample_similarities.append(similarity)
    
    return {
        'overall_correlation': overall_corr,
        'overall_p_value': overall_p,
        'dimension_correlations': dim_correlations,
        'sample_similarities': sample_similarities,
        'mean_similarity': np.mean(sample_similarities),
        'std_similarity': np.std(sample_similarities)
    }
```

## Distance Metrics

### Intra-class vs Inter-class Distances

```python
def compute_class_distance_metrics(embeddings, labels, distance_metric='euclidean'):
    """Compute intra-class and inter-class distance statistics."""
    
    from scipy.spatial.distance import pdist, cdist
    
    unique_labels = list(set(labels))
    intra_distances = []
    inter_distances = []
    
    # Compute intra-class distances
    for label in unique_labels:
        label_mask = np.array(labels) == label
        label_embeddings = embeddings[label_mask]
        
        if len(label_embeddings) > 1:
            distances = pdist(label_embeddings, metric=distance_metric)
            intra_distances.extend(distances)
    
    # Compute inter-class distances
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j:  # Avoid duplicates
                mask1 = np.array(labels) == label1
                mask2 = np.array(labels) == label2
                
                emb1 = embeddings[mask1]
                emb2 = embeddings[mask2]
                
                distances = cdist(emb1, emb2, metric=distance_metric)
                inter_distances.extend(distances.flatten())
    
    # Calculate statistics
    intra_stats = {
        'mean': np.mean(intra_distances) if intra_distances else 0,
        'std': np.std(intra_distances) if intra_distances else 0,
        'median': np.median(intra_distances) if intra_distances else 0,
        'min': np.min(intra_distances) if intra_distances else 0,
        'max': np.max(intra_distances) if intra_distances else 0
    }
    
    inter_stats = {
        'mean': np.mean(inter_distances) if inter_distances else 0,
        'std': np.std(inter_distances) if inter_distances else 0,
        'median': np.median(inter_distances) if inter_distances else 0,
        'min': np.min(inter_distances) if inter_distances else 0,
        'max': np.max(inter_distances) if inter_distances else 0
    }
    
    # Separation ratio
    separation_ratio = (inter_stats['mean'] / intra_stats['mean'] 
                       if intra_stats['mean'] > 0 else float('inf'))
    
    return {
        'intra_class_distances': intra_stats,
        'inter_class_distances': inter_stats,
        'separation_ratio': separation_ratio,
        'raw_intra_distances': intra_distances,
        'raw_inter_distances': inter_distances
    }
```

## Hypothesis Testing

### Species Clustering Significance

```python
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def test_species_clustering_significance(embeddings, species_labels, n_clusters=None):
    """Test statistical significance of species clustering."""
    
    from sklearn.cluster import KMeans
    
    if n_clusters is None:
        n_clusters = len(set(species_labels))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Create contingency table
    species_list = sorted(list(set(species_labels)))
    cluster_list = sorted(list(set(cluster_labels)))
    
    contingency_table = np.zeros((len(species_list), len(cluster_list)))
    
    for i, species in enumerate(species_list):
        for j, cluster in enumerate(cluster_list):
            count = sum((np.array(species_labels) == species) & 
                       (cluster_labels == cluster))
            contingency_table[i, j] = count
    
    # Chi-square test
    chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
    
    # Clustering metrics
    ari_score = adjusted_rand_score(species_labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(species_labels, cluster_labels)
    
    return {
        'contingency_table': contingency_table,
        'chi2_statistic': chi2_stat,
        'chi2_p_value': chi2_p,
        'degrees_of_freedom': dof,
        'adjusted_rand_index': ari_score,
        'normalized_mutual_info': nmi_score,
        'significant_clustering': chi2_p < 0.05
    }
```

### Permutation Tests

```python
def permutation_test_embedding_structure(embeddings, labels, n_permutations=1000, 
                                       metric='silhouette'):
    """Test if embedding structure is better than random using permutation test."""
    
    # Compute observed metric
    if metric == 'silhouette':
        observed_score = silhouette_score(embeddings, labels)
    elif metric == 'calinski_harabasz':
        observed_score = calinski_harabasz_score(embeddings, labels)
    elif metric == 'davies_bouldin':
        observed_score = davies_bouldin_score(embeddings, labels)
    
    # Permutation test
    permuted_scores = []
    
    for _ in range(n_permutations):
        # Shuffle labels
        permuted_labels = np.random.permutation(labels)
        
        # Compute metric with shuffled labels
        if metric == 'silhouette':
            score = silhouette_score(embeddings, permuted_labels)
        elif metric == 'calinski_harabasz':
            score = calinski_harabasz_score(embeddings, permuted_labels)
        elif metric == 'davies_bouldin':
            score = davies_bouldin_score(embeddings, permuted_labels)
        
        permuted_scores.append(score)
    
    permuted_scores = np.array(permuted_scores)
    
    # Calculate p-value
    if metric in ['silhouette', 'calinski_harabasz']:  # Higher is better
        p_value = np.mean(permuted_scores >= observed_score)
    else:  # Lower is better (davies_bouldin)
        p_value = np.mean(permuted_scores <= observed_score)
    
    return {
        'observed_score': observed_score,
        'permuted_scores': permuted_scores,
        'mean_permuted_score': np.mean(permuted_scores),
        'std_permuted_score': np.std(permuted_scores),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': (observed_score - np.mean(permuted_scores)) / np.std(permuted_scores)
    }
```

## Biological Relevance Metrics

### Phylogenetic Signal

```python
def compute_phylogenetic_signal(embeddings, phylogenetic_distances):
    """Compute phylogenetic signal in embeddings using Mantel test."""
    
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    
    # Compute embedding distances
    embedding_distances = pdist(embeddings, metric='euclidean')
    
    # Flatten phylogenetic distance matrix if needed
    if phylogenetic_distances.ndim == 2:
        phylo_distances = squareform(phylogenetic_distances, checks=False)
    else:
        phylo_distances = phylogenetic_distances
    
    # Mantel test (correlation between distance matrices)
    correlation, p_value = pearsonr(embedding_distances, phylo_distances)
    
    return {
        'mantel_correlation': correlation,
        'mantel_p_value': p_value,
        'phylogenetic_signal': correlation > 0 and p_value < 0.05
    }
```

### Morphological Correlation

```python
def compute_morphological_correlations(embeddings, morphological_features):
    """Compute correlations with morphological measurements."""
    
    correlations = {}
    
    for feature_name, feature_values in morphological_features.items():
        # Compute correlation with each embedding dimension
        dim_correlations = []
        
        for dim in range(embeddings.shape[1]):
            corr, p_val = pearsonr(embeddings[:, dim], feature_values)
            dim_correlations.append({
                'dimension': dim,
                'correlation': corr,
                'p_value': p_val
            })
        
        # Find dimension with highest absolute correlation
        best_dim = max(dim_correlations, key=lambda x: abs(x['correlation']))
        
        correlations[feature_name] = {
            'all_dimensions': dim_correlations,
            'best_dimension': best_dim,
            'max_correlation': best_dim['correlation']
        }
    
    return correlations
```

## Model Performance Metrics

### Embedding Quality Score

```python
def compute_embedding_quality_score(embeddings, labels, weights=None):
    """Compute composite embedding quality score."""
    
    if weights is None:
        weights = {
            'silhouette': 0.3,
            'calinski_harabasz': 0.3,
            'davies_bouldin': 0.2,
            'separation_ratio': 0.2
        }
    
    # Compute individual metrics
    silhouette = silhouette_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    # Distance metrics
    distance_metrics = compute_class_distance_metrics(embeddings, labels)
    separation_ratio = distance_metrics['separation_ratio']
    
    # Normalize metrics (0-1 scale)
    normalized_silhouette = (silhouette + 1) / 2  # [-1, 1] -> [0, 1]
    normalized_calinski = min(calinski_harabasz / 1000, 1)  # Cap at 1000
    normalized_davies = max(0, 1 - davies_bouldin / 10)  # Lower is better
    normalized_separation = min(separation_ratio / 10, 1)  # Cap at 10
    
    # Compute weighted score
    quality_score = (
        weights['silhouette'] * normalized_silhouette +
        weights['calinski_harabasz'] * normalized_calinski +
        weights['davies_bouldin'] * normalized_davies +
        weights['separation_ratio'] * normalized_separation
    )
    
    return {
        'quality_score': quality_score,
        'components': {
            'silhouette': normalized_silhouette,
            'calinski_harabasz': normalized_calinski,
            'davies_bouldin': normalized_davies,
            'separation_ratio': normalized_separation
        },
        'raw_metrics': {
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin,
            'separation_ratio': separation_ratio
        }
    }
```

## Statistical Analysis Pipeline

### Comprehensive Statistical Report

```python
def generate_statistical_report(embeddings, labels, metadata_df=None, 
                              output_dir='statistical_analysis'):
    """Generate comprehensive statistical analysis report."""
    
    from pathlib import Path
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    report = {}
    
    # 1. Clustering metrics
    print("Computing clustering metrics...")
    report['silhouette_analysis'] = compute_silhouette_analysis(embeddings, labels)
    report['calinski_harabasz'] = compute_calinski_harabasz_score(embeddings, labels)
    report['davies_bouldin'] = compute_davies_bouldin_score(embeddings, labels)
    
    # 2. Distance metrics
    print("Computing distance metrics...")
    report['distance_metrics'] = compute_class_distance_metrics(embeddings, labels)
    
    # 3. Embedding quality score
    print("Computing quality score...")
    report['quality_score'] = compute_embedding_quality_score(embeddings, labels)
    
    # 4. Metadata correlations (if available)
    if metadata_df is not None:
        print("Computing metadata correlations...")
        correlations = compute_embedding_metadata_correlations(embeddings, metadata_df)
        report['metadata_correlations'] = correlations
        report['significant_correlations'] = find_significant_correlations(correlations)
    
    # 5. Permutation tests
    print("Running permutation tests...")
    report['permutation_tests'] = {
        'silhouette': permutation_test_embedding_structure(embeddings, labels, 
                                                         metric='silhouette'),
        'calinski_harabasz': permutation_test_embedding_structure(embeddings, labels, 
                                                                metric='calinski_harabasz')
    }
    
    # 6. Species clustering significance
    print("Testing species clustering significance...")
    report['species_clustering'] = test_species_clustering_significance(embeddings, labels)
    
    # Save report
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    report_json = convert_numpy(report)
    
    with open(output_dir / 'statistical_report.json', 'w') as f:
        json.dump(report_json, f, indent=2)
    
    # Create summary
    summary = {
        'n_samples': len(embeddings),
        'n_dimensions': embeddings.shape[1],
        'n_classes': len(set(labels)),
        'overall_quality_score': report['quality_score']['quality_score'],
        'silhouette_score': report['silhouette_analysis']['overall_silhouette'],
        'significant_clustering': report['species_clustering']['significant_clustering']
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Statistical analysis complete. Results saved to {output_dir}")
    
    return report, summary
```

## Next Steps

- **[Embedding Analysis](embedding-analysis.md)**: Advanced embedding analysis techniques  
- **[Visualization Tools](visualization.md)**: Visual analysis methods
- **[Examples](../examples/)**: Practical statistical analysis examples
