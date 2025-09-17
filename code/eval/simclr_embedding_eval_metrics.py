import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples, r2_score
import os
import json
import argparse
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


def calculate_silhouette_metrics(embeddings, labels, metric='euclidean', **metric_params):
    """
    Calculate silhouette score.
    Args:
        embeddings (np.ndarray): The embedding vectors
        labels (pd.Series): Species labels
        metric (str): The distance metric to use
        **metric_params: Additional parameters for the metric (e.g., VI for mahalanobis -- not used at this stage)
        
    Returns:
        dict: Dictionary containing overall and per-species silhouette scores
    """
    # Calculate silhouette scores
    sample_scores = silhouette_samples(
        X=embeddings,
        labels=labels,
        metric=metric,
        **metric_params
    )
    
    # Calculate per-species mean silhouette scores
    per_species_scores = {
        species: float(sample_scores[labels == species].mean())
        for species in np.unique(labels)
    }
    
    # Calculate overall mean silhouette score
    overall_score = float(sample_scores.mean())
    
    return {
        'overall_silhouette_score': overall_score,
        'per_species_silhouette_scores': per_species_scores
    }


def calculate_r_squared(embeddings, labels):
    """
    Calculate R-squared score using PCA to measure how well species explain the variance.
    Args:
        embeddings (np.ndarray): The embedding vectors
        labels (np.Series): Species labels
        
    Returns:
        dict: Dictionary containing R-squared scores for different numbers of components
    """
    # Try different numbers of components
    n_components_list = [2, 3, 5, 10]
    r2_scores = {}
    
    for n_comp in n_components_list:
        # Perform PCA
        pca = PCA(n_components=n_comp)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Calculate R-squared for each component
        r2_per_component = []
        unique_labels = np.unique(labels)
        label_means = np.array([reduced_embeddings[labels == label].mean(axis=0) 
                              for label in unique_labels])
        
        for i in range(n_comp):
            # Create predictions by assigning the species mean to each sample
            predictions = np.zeros(len(embeddings))
            for j, label in enumerate(unique_labels):
                mask = labels == label
                predictions[mask] = label_means[j, i]
            
            # Calculate R-squared for this component
            r2 = r2_score(reduced_embeddings[:, i], predictions)
            r2_per_component.append(float(r2))
        
        r2_scores[f'pca_{n_comp}_components'] = {
            'per_component': r2_per_component,
            'mean': float(np.mean(r2_per_component))
        }
    
    return r2_scores


def calculate_r_squared_anova(distance_matrix, labels):
    """
    Calculate R² value using ANOVA approach with distance matrix and labels.
    Args:
        distance_matrix (np.ndarray): Square matrix of pairwise distances
        labels (np.Series): Series containing group (species) labels
        
    Returns:
        float: R² value representing proportion of variance explained
    """
    # Calculate total sum of squares
    total_mean = np.mean(distance_matrix)
    SS_total = np.sum((distance_matrix - total_mean) ** 2) / 2  # Divide by 2 to not count each pair twice
    
    # Calculate between-group sum of squares
    unique_groups = np.unique(labels)
    SS_between = 0
    n_total = len(labels)
    
    for i, group1 in enumerate(unique_groups):
        mask1 = labels == group1
        n1 = np.sum(mask1)
        
        for group2 in unique_groups[i+1:]:
            mask2 = labels == group2
            n2 = np.sum(mask2)
            
            # Get between-group distances
            between_distances = distance_matrix[np.ix_(mask1, mask2)]
            group_mean = np.mean(between_distances)
            
            # Weight by number of samples in both groups
            weight = (n1 * n2) / n_total
            SS_between += weight * (group_mean - total_mean) ** 2
    
    # Calculate R²
    r_squared = float(SS_between / SS_total)
    
    return r_squared


def extract_metadata_from_filename(filename):
    """
    Extract species and sex from filename.
    Example: "Acrocephalus_atyphus_1_M_Back.png" -> species="Acrocephalus_atyphus", sex="M"
    """
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    
    # print(f"Filename: {base_name}")
    # print(f"Parts: {parts}")
    
    if len(parts) >= 4:  # Should have at least: genus_species_number_sex
        species = f"{parts[0]}_{parts[1]}"  # Combine genus and species
        sex = parts[3] if parts[3] in ['M', 'F', 'U'] else None  # Get sex if it's M or F or U
        # print(f"Extracted: species={species}, sex={sex}")  # Debug print
        return species, sex
    
    # print(f"Not enough parts in filename: {base_name}")  # Debug print
    return base_name, None


def calculate_variance_explained(embeddings, labels, n_components=None):
    """
    Calculate variance explained by group labels using PCA.
    Args:
        embeddings (np.ndarray): The embedding vectors
        labels (np.array): Group labels
        n_components (int): Number of PCA components to use
        
    Returns:
        dict: Dictionary containing variance explained metrics
    """
    # Perform PCA
    if n_components is None:
        n_components = min(embeddings.shape[0], embeddings.shape[1])
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embeddings)
    
    # Calculate total variance explained by PCA components
    total_variance_ratio = pca.explained_variance_ratio_
    
    # For each component, calculate how much variance is explained by the grouping
    variance_explained = []
    for component_idx in range(n_components):
        component_values = pca_result[:, component_idx]
        
        # Calculate total variance in this component
        total_var = np.var(component_values)
        
        # Calculate between-group variance
        group_means = {group: np.mean(component_values[labels == group]) 
                      for group in np.unique(labels)}
        
        between_var = np.var(list(group_means.values()))
        
        # Calculate variance explained ratio
        var_explained = between_var / total_var if total_var > 0 else 0
        variance_explained.append(float(var_explained))
    
    return {
        'per_component_variance_explained': variance_explained,
        'pca_explained_variance_ratio': total_variance_ratio.tolist(),
        'weighted_mean_variance_explained': float(np.average(
            variance_explained, 
            weights=total_variance_ratio[:len(variance_explained)]
        ))
    }


def calculate_manova_variance_explained(embeddings, labels):
    """
    Calculate variance explained using MANOVA approach.
    
    Args:
        embeddings (np.ndarray): The embedding vectors
        labels (np.array): Group labels
        
    Returns:
        float: Proportion of variance explained (Wilks' Lambda transformed to R²-like metric)
    """
    # Calculate total sum of squares matrix
    grand_mean = embeddings.mean(axis=0)
    SS_total = np.zeros((embeddings.shape[1], embeddings.shape[1]))
    for i in range(len(embeddings)):
        dev = embeddings[i] - grand_mean
        SS_total += np.outer(dev, dev)
    
    # Calculate between-groups sum of squares matrix
    SS_between = np.zeros_like(SS_total)
    unique_groups = np.unique(labels)
    
    for group in unique_groups:
        group_mask = labels == group
        group_size = np.sum(group_mask)
        if group_size > 0:
            group_mean = embeddings[group_mask].mean(axis=0)
            dev = group_mean - grand_mean
            SS_between += group_size * np.outer(dev, dev)
    
    # Calculate within-groups sum of squares matrix
    SS_within = SS_total - SS_between
    
    # Calculate total variance
    total_var = np.trace(SS_total)
    
    # Calculate between-groups variance
    between_var = np.trace(SS_between)
    
    # Calculate variance explained ratio (similar to R²)
    var_explained = float(between_var / total_var) if total_var > 0 else 0.0
    
    return {
        'variance_explained': var_explained,
        'total_variance': float(total_var),
        'between_variance': float(between_var),
        'within_variance': float(np.trace(SS_within))
    }



def analyze_embeddings(embeddings_csv, output_dir, batch_size=1000, num_samples=None, random_seed=42, calculate_r_squared=False):
    """
    Analyze SimCLR embeddings and calculate various metrics.
    Args:
        embeddings_csv (str): Path to CSV file containing embeddings
        output_dir (str): Directory to save results
        batch_size (int): Size of batches for processing large datasets
        num_samples (int, optional): Number of random samples to analyze. If None, use full dataset
        random_seed (int): Random seed for reproducibility
        calculate_r_squared (bool): If True, also calculate r-squared score in addition to silhouette scores.
    """
    # Read embeddings
    print("Reading embeddings...")
    df = pd.read_csv(embeddings_csv)
    
    # Sample random subset if num_samples is specified
    if num_samples is not None and num_samples < len(df):
        print(f"\nSampling {num_samples} random samples from dataset of size {len(df)}...")
        df = df.sample(n=num_samples, random_state=random_seed)
        print("Sampling complete.")
    
    filenames = df['filename'].values
    embeddings = df.drop('filename', axis=1).values
    
    # Extract species and sex from filenames
    print("Extracting metadata...")
    species_labels = []
    sex_labels = []
    for filename in filenames:
        species, sex = extract_metadata_from_filename(filename)
        # print(species, sex)
        species_labels.append(species)
        sex_labels.append(sex)
    
    species_labels = np.array(species_labels)
    sex_labels = np.array(sex_labels)
    
    # Normalize embeddings
    print("\nNormalizing embeddings...")
    normalized_embeddings = normalize(embeddings)
    
    results = {
        'metadata': {
            'num_samples': len(embeddings),
            'num_species': len(np.unique(species_labels)),
            'num_sexes': len(np.unique(sex_labels)),
            'embedding_dim': embeddings.shape[1],
            'random_seed': random_seed if num_samples is not None else None,
            'sampled_data': num_samples is not None
        }
    }

    # metrics for silhouette analysis
    metrics = ['euclidean', 'cosine', 'correlation']
    
    # Calculate silhouette scores for each metric
    print("\nCalculating silhouette scores...")
    silhouette_results = {}
    
    # For Mahalanobis distance, compute covariance matrix
    mahalanobis_params = {}
    if 'mahalanobis' in metrics:  # List of metrics to use
        cov = np.cov(normalized_embeddings.T)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            inv_cov = np.linalg.pinv(cov)
        mahalanobis_params = {'VI': inv_cov}
    
    for metric in metrics:
        print(f"\nCalculating {metric} silhouette scores...")
        metric_params = mahalanobis_params if metric == 'mahalanobis' else {}
        metric_results = calculate_silhouette_metrics(
            normalized_embeddings, 
            species_labels,
            metric=metric,
            **metric_params
        )
        silhouette_results[f'{metric}_distance'] = metric_results
    
    results['silhouette_analysis'] = silhouette_results
    
    # Calculate R-squared if requested
    if calculate_r_squared:
        print("\nCalculating R-squared score using ANOVA approach...")
        # For R-squared, we'll use Euclidean distance
        euclidean_distances = pdist(normalized_embeddings, metric='euclidean')
        distance_matrix = squareform(euclidean_distances)
        r2_anova = calculate_r_squared_anova(distance_matrix, species_labels)
        
        results['r_squared_analysis'] = {
            'anova_based': {
                'r_squared': r2_anova,
                'description': 'R-squared calculated using ANOVA sum of squares approach on distance matrix'
            }
        }
        
        print(f"\nR-squared (ANOVA): {r2_anova:.4f}")
    
    print("\nSilhouette Score Summary:")
    for metric in metrics:
        print(f"Overall silhouette score ({metric}): {silhouette_results[f'{metric}_distance']['overall_silhouette_score']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'embedding_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze SimCLR embeddings')
    parser.add_argument('--embeddings', help='Path to CSV file containing embeddings')
    parser.add_argument('--output', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=1000, help='Size of batches for processing large datasets')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of random samples to analyze')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--r-squared', action='store_true', help='Calculate R-squared score in addition to silhouette scores')
    
    args = parser.parse_args()
    
    analyze_embeddings(
        args.embeddings,
        args.output,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        calculate_r_squared=args.r_squared
    )

if __name__ == "__main__":
    main()