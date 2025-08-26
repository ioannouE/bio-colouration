import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples
import os
from skbio.stats.distance import DistanceMatrix, permanova
import argparse
import re

def calculate_within_species_variance(distance_matrix, metadata):
    """
    Calculate the average within-species variance using the distance matrix.
    
    Args:
        distance_matrix (np.ndarray): Square matrix of pairwise distances
        metadata (pd.DataFrame): DataFrame containing species information
        
    Returns:
        pd.Series: Series containing variance for each species
    """
    within_var = {}
    for species in metadata['species'].unique():
        # Get indices for this species
        species_mask = metadata['species'] == species
        species_indices = np.where(species_mask)[0]
        
        if len(species_indices) > 1:  # Need at least 2 samples to calculate variance
            # Get distances for this species
            species_distances = distance_matrix[np.ix_(species_indices, species_indices)]
            # Calculate variance (excluding diagonal) - np.triu returns the upper triangular part of a matrix, excluding the diagonal
            upper_tri = np.triu(species_distances, k=1)
            variance = np.mean(upper_tri[upper_tri != 0])
            within_var[species] = variance
    
    return pd.Series(within_var)



def calculate_between_species_variance(distance_matrix, metadata):
    """
    Calculate the variance between different species using the distance matrix.
    
    Args:
        distance_matrix (np.ndarray): Square matrix of pairwise distances
        metadata (pd.DataFrame): DataFrame containing species information
        
    Returns:
        pd.Series: Series containing pairwise variance between species
    """
    species_list = metadata['species'].unique()
    between_var = {}
    
    for i, species1 in enumerate(species_list):
        for species2 in species_list[i+1:]:
            # Get indices for both species
            mask1 = metadata['species'] == species1
            mask2 = metadata['species'] == species2
            indices1 = np.where(mask1)[0]
            indices2 = np.where(mask2)[0]
            
            # Get distances between the two species
            between_distances = distance_matrix[np.ix_(indices1, indices2)]
            # Calculate the mean of the distances
            mean_dist = np.mean(between_distances)
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
    # overall_score_direct = silhouette_score(
    #     X=distance_matrix,
    #     labels=labels,
    #     metric='precomputed'
    # )

    # print(f"Overall silhouette score: {overall_score_direct:.6f}")

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

def perform_simclr_permanova(embeddings_csv, output_dir):
    """
    Perform PERMANOVA analysis on SimCLR embeddings and calculate various metrics.
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
    
    # Calculate within and between species variance
    print("\nCalculating within-species variance...")
    within_var = calculate_within_species_variance(distance_matrix, metadata)
    print("\nCalculating between-species variance...")
    between_var = calculate_between_species_variance(distance_matrix, metadata)
    
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
        permutations=999
    )
    
    # Then with "euclidified" distances (square root transformation)
    print("Performing PERMANOVA with euclidified distances...")
    euclidified_distances = np.sqrt(distance_matrix)
    euclidified_matrix_skbio = DistanceMatrix(euclidified_distances, ids=metadata.index)
    permanova_results_euclidified = permanova(
        euclidified_matrix_skbio,
        grouping=metadata['species'],
        permutations=999
    )
    
    # Save results
    results = {
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
    variance_dir = os.path.join(output_dir, 'variance_analysis')
    os.makedirs(variance_dir, exist_ok=True)
    
    within_var.to_csv(os.path.join(variance_dir, 'within_species_variance.csv'))
    between_var.to_csv(os.path.join(variance_dir, 'between_species_variance.csv'))
    
    # Save silhouette scores
    silhouette_dir = os.path.join(output_dir, 'silhouette_analysis')
    os.makedirs(silhouette_dir, exist_ok=True)
    
    # Save per-species silhouette scores
    species_silhouette.to_csv(os.path.join(silhouette_dir, 'species_silhouette_scores.csv'))
    
    # Calculate and save summary statistics
    variance_summary = {
        'mean_within_species_variance': within_var.mean(),
        'std_within_species_variance': within_var.std(),
        'mean_between_species_variance': between_var.mean(),
        'std_between_species_variance': between_var.std(),
        'variance_ratio': within_var.mean() / between_var.mean(),
        'overall_silhouette_score': overall_silhouette
    }
    
    pd.Series(variance_summary).to_csv(os.path.join(variance_dir, 'variance_summary.csv'))
    
    # Print results
    print("\nPERMANOVA Results:")
    print("------------------")
    print("\nRegular PERMANOVA:")
    print(f"Test statistic: {results['Regular PERMANOVA']['test statistic']:.6f}")
    print(f"p-value: {results['Regular PERMANOVA']['p-value']:.6f}")
    
    print("\nVariance Analysis:")
    print("------------------")
    print(f"Mean within-species variance: {variance_summary['mean_within_species_variance']:.6f}")
    print(f"Mean between-species variance: {variance_summary['mean_between_species_variance']:.6f}")
    print(f"Variance ratio (within/between): {variance_summary['variance_ratio']:.6f}")
    
    print("\nSilhouette Analysis:")
    print("--------------------")
    print(f"Overall silhouette score: {overall_silhouette:.6f}")
    print("\nTop 5 species by silhouette score:")
    print(species_silhouette.sort_values(ascending=False).head())
    print("\nBottom 5 species by silhouette score:")
    print(species_silhouette.sort_values(ascending=False).tail())
    
    print(f"\nDetailed results saved in: {variance_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate SimCLR embeddings with PERMANOVA and variance analysis')
    parser.add_argument('--embeddings', type=str, required=True,
                      help='Path to embeddings CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    perform_simclr_permanova(args.embeddings, args.output_dir)

if __name__ == "__main__":
    main()