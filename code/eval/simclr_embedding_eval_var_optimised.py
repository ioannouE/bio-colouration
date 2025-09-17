import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples
import os
from skbio.stats.distance import DistanceMatrix, permanova
import argparse
import re
from numba import jit, prange
import numba as nb

@jit(nopython=True, parallel=True)
def _calculate_within_species_distances(distance_matrix, species_indices):
    """
    Helper function to calculate within-species distances using Numba.
    """
    n = len(species_indices)
    distances = np.zeros((n * (n-1)) // 2)
    idx = 0
    for i in prange(n):
        for j in range(i+1, n):
            distances[idx] = distance_matrix[species_indices[i], species_indices[j]]
            idx += 1
    return np.mean(distances) if len(distances) > 0 else 0.0

@jit(nopython=True, parallel=True)
def _calculate_between_species_distances(distance_matrix, indices1, indices2):
    """
    Helper function to calculate between-species distances using Numba.
    """
    n1, n2 = len(indices1), len(indices2)
    total = 0.0
    count = 0
    for i in prange(n1):
        for j in range(n2):
            total += distance_matrix[indices1[i], indices2[j]]
            count += 1
    return total / count if count > 0 else 0.0

def calculate_within_species_variance(distance_matrix, metadata, max_samples_per_species=100, seed=42):
    """
    Calculate the average within-species variance using the distance matrix with sampling.
    
    Args:
        distance_matrix (np.ndarray): Square matrix of pairwise distances
        metadata (pd.DataFrame): DataFrame containing species information
        max_samples_per_species (int): Maximum number of samples to use per species
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.Series: Series containing variance for each species
    """
    np.random.seed(seed)
    within_var = {}
    species_categories = pd.Categorical(metadata['species'])
    species_codes = species_categories.codes
    unique_species = species_categories.categories
    
    for species_idx, species in enumerate(unique_species):
        species_indices = np.where(species_codes == species_idx)[0]
        if len(species_indices) > 1:
            # Sample indices if there are more than max_samples_per_species
            if len(species_indices) > max_samples_per_species:
                species_indices = np.random.choice(
                    species_indices, 
                    size=max_samples_per_species, 
                    replace=False
                )
            
            variance = _calculate_within_species_distances(
                distance_matrix,
                species_indices.astype(np.int64)
            )
            within_var[species] = variance
    
    return pd.Series(within_var)

def calculate_between_species_variance(distance_matrix, metadata, max_samples_per_species=100, max_species_pairs=None, seed=42):
    """
    Calculate the variance between different species using the distance matrix with sampling.
    
    Args:
        distance_matrix (np.ndarray): Square matrix of pairwise distances
        metadata (pd.DataFrame): DataFrame containing species information
        max_samples_per_species (int): Maximum number of samples to use per species
        max_species_pairs (int): Maximum number of species pairs to compare. If None, compare all pairs
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.Series: Series containing pairwise variance between species
    """
    np.random.seed(seed)
    species_categories = pd.Categorical(metadata['species'])
    species_codes = species_categories.codes
    species_list = species_categories.categories
    between_var = {}
    
    # Create all possible species pairs
    species_pairs = []
    for i, species1 in enumerate(species_list):
        for j, species2 in enumerate(species_list[i+1:], start=i+1):
            species_pairs.append((i, j, species1, species2))
    
    # Sample species pairs if requested
    if max_species_pairs is not None and len(species_pairs) > max_species_pairs:
        species_pairs = np.random.choice(
            species_pairs, 
            size=max_species_pairs, 
            replace=False
        )
    
    # Calculate distances for selected pairs
    for i, j, species1, species2 in species_pairs:
        indices1 = np.where(species_codes == i)[0]
        indices2 = np.where(species_codes == j)[0]
        
        # Sample indices if there are more than max_samples_per_species
        if len(indices1) > max_samples_per_species:
            indices1 = np.random.choice(indices1, size=max_samples_per_species, replace=False)
        if len(indices2) > max_samples_per_species:
            indices2 = np.random.choice(indices2, size=max_samples_per_species, replace=False)
        
        mean_dist = _calculate_between_species_distances(
            distance_matrix,
            indices1.astype(np.int64),
            indices2.astype(np.int64)
        )
        between_var[f"{species1}_vs_{species2}"] = mean_dist
    
    return pd.Series(between_var)

def calculate_silhouette_metrics(distance_matrix, labels):
    """
    Calculate silhouette score and per-species silhouette scores.
    
    Args:
        distance_matrix (np.ndarray): Precomputed distance matrix
        labels (pd.Series): Species labels
        
    Returns:
        tuple: (overall_score, per_species_scores)
            - overall_score: float, silhouette score for entire dataset
            - per_species_scores: pd.Series, mean silhouette score per species
    """
    # Calculate silhouette scores for each sample
    sample_scores = silhouette_samples(
        X=distance_matrix,
        labels=labels,
        metric='precomputed'  # Important: we're using pre-computed distances
    )
    
    # Calculate per-species mean silhouette scores
    per_species_scores = pd.Series(
        {species: sample_scores[labels == species].mean() 
         for species in np.unique(labels)}
    )
    
    # Calculate overall mean silhouette score
    overall_score = sample_scores.mean()
    
    print("Silhouette scores:")
    print(f"Overall silhouette score: {overall_score:.6f}")
    print("Per-species silhouette scores:")
    print(per_species_scores)
    
    return overall_score, per_species_scores

def perform_simclr_permanova(embeddings_csv, output_dir, max_samples_per_species=100, max_species_pairs=None):
    """
    Perform PERMANOVA analysis on SimCLR embeddings and calculate various metrics.
    
    Args:
        embeddings_csv (str): Path to the embeddings CSV file
        output_dir (str): Directory to save results
        max_samples_per_species (int): Maximum number of samples to use per species
        max_species_pairs (int): Maximum number of species pairs to compare
    """
    # Read embeddings
    print("Reading embeddings...")
    df = pd.read_csv(embeddings_csv)
    filenames = df['filename'].values
    embeddings = df.drop('filename', axis=1).values
    
    # Extract species names from filenames and create metadata DataFrame
    print("Extracting species names...")
    def extract_species(filename):
        base_name = os.path.basename(filename)
        parts = base_name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return base_name
    
    metadata = pd.DataFrame({
        'filename': filenames,
        'species': [extract_species(f) for f in filenames]
    })
    metadata.set_index('filename', inplace=True)
    
    # Print species counts before proceeding
    print("\nSpecies counts:")
    species_counts = metadata['species'].value_counts()
    print(species_counts)
    
    # Calculate distance matrix using cosine distance
    print("\nCalculating distance matrix...")
    normalized_embeddings = normalize(embeddings)
    distances = pdist(normalized_embeddings, metric='cosine')
    distance_matrix = squareform(distances)
    
    # Calculate within and between species variance with sampling
    print("\nCalculating within-species variance...")
    within_var = calculate_within_species_variance(
        distance_matrix, 
        metadata,
        max_samples_per_species=max_samples_per_species
    )
    
    print("\nCalculating between-species variance...")
    between_var = calculate_between_species_variance(
        distance_matrix, 
        metadata,
        max_samples_per_species=max_samples_per_species,
        max_species_pairs=max_species_pairs
    )
    
    # Calculate silhouette scores
    print("\nCalculating silhouette scores...")
    overall_silhouette, species_silhouette = calculate_silhouette_metrics(
        distance_matrix, 
        metadata['species']
    )
    
    # Convert to skbio DistanceMatrix using filenames as IDs
    distance_matrix_skbio = DistanceMatrix(distance_matrix, ids=metadata.index)
    
    # Perform PERMANOVA
    print("Performing PERMANOVA...")
    permanova_results = permanova(
        distance_matrix_skbio,
        grouping=metadata['species'],
        permutations=99
    )
    
    # Then with "euclidified" distances (square root transformation)
    print("Performing PERMANOVA with euclidified distances...")
    euclidified_distances = np.sqrt(distance_matrix)
    euclidified_matrix_skbio = DistanceMatrix(euclidified_distances, ids=metadata.index)
    permanova_results_euclidified = permanova(
        euclidified_matrix_skbio,
        grouping=metadata['species'],
        permutations=99
    )
    
    # Save results
    results = {
        'Silhouette scores': {
            'Overall silhouette score': overall_silhouette,
            'Per-species silhouette scores': species_silhouette
        },
        'Regular PERMANOVA': {
            'test statistic': permanova_results['test statistic'].item(),
            'p-value': permanova_results['p-value'].item(),
            'number of permutations': permanova_results['number of permutations']
        },
        'Euclidified PERMANOVA': {
            'test statistic': permanova_results_euclidified['test statistic'].item(),
            'p-value': permanova_results_euclidified['p-value'].item(),
            'number of permutations': permanova_results_euclidified['number of permutations']
        }
    }
    
    # Save all results
    results_df = pd.DataFrame(results).round(6)
    output_file = os.path.join(output_dir, 'permanova_results.csv')
    results_df.to_csv(output_file)
    print(f"\nPERMANOVA results saved to: {output_file}")
    
    # Save variance results
    within_var.to_csv(os.path.join(output_dir, 'within_species_variance.csv'))
    between_var.to_csv(os.path.join(output_dir, 'between_species_variance.csv'))

def main():
    parser = argparse.ArgumentParser(description='Perform PERMANOVA analysis on SimCLR embeddings')
    parser.add_argument('--embeddings', help='Path to embeddings CSV file')
    parser.add_argument('--output-dir', help='Directory to save results')
    parser.add_argument('--max-samples', type=int, default=100,
                      help='Maximum number of samples to use per species (default: 100)')
    parser.add_argument('--max-pairs', type=int, default=None,
                      help='Maximum number of species pairs to compare (default: None, compare all)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    perform_simclr_permanova(
        args.embeddings, 
        args.output_dir,
        max_samples_per_species=args.max_samples,
        max_species_pairs=args.max_pairs
    )

if __name__ == "__main__":
    main()