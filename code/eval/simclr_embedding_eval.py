import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
import os
from skbio.stats.distance import DistanceMatrix, permanova
import re

def calculate_r_squared(distance_matrix, grouping):
    """
    Calculate R² value from distance matrix and grouping information.
    
    Args:
        distance_matrix (np.ndarray): Square matrix of pairwise distances
        grouping (pd.Series): Series containing group labels
        
    Returns:
        float: R² value representing proportion of variance explained
    """
    n = len(grouping)
    
    # Calculate total sum of squares
    total_mean = np.mean(distance_matrix)
    SS_total = np.sum((distance_matrix - total_mean) ** 2) / 2  # Divide by 2 to not count each pair twice
    
    # Calculate between-group sum of squares
    unique_groups = np.unique(grouping)
    SS_between = 0
    
    for group in unique_groups:
        group_mask = grouping == group
        if np.sum(group_mask) > 1:  # Only consider groups with more than one sample
            group_distances = distance_matrix[np.ix_(group_mask, group_mask)]
            group_mean = np.mean(group_distances)
            SS_between += np.sum((group_mean - total_mean) ** 2)
    
    # Calculate R²
    r_squared = SS_between / SS_total
    
    return r_squared

def perform_simclr_permanova(embeddings_csv, output_dir):
    """
    Perform PERMANOVA analysis on SimCLR embeddings to test if within-species variance
    is less than between-species variance.
    """
    # Read embeddings
    print("Reading embeddings...")
    df = pd.read_csv(embeddings_csv)
    filenames = df['filename'].values
    embeddings = df.drop('filename', axis=1).values
    
    # Extract species names from filenames and create metadata DataFrame
    print("Extracting species names...")
    def extract_species(filename):
        # Get just the filename without path
        base_name = os.path.basename(filename)
        # Split by underscore and take first two parts
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
    
    # Convert to skbio DistanceMatrix using filenames as IDs
    distance_matrix_skbio = DistanceMatrix(distance_matrix, ids=metadata.index)
    
    # Perform PERMANOVA and calculate R²
    print("Performing PERMANOVA...")
    permanova_results = permanova(
        distance_matrix_skbio,
        grouping=metadata['species'],
        permutations=999
    )
    
    r_squared = calculate_r_squared(distance_matrix, metadata['species'])
    
    # Then with "euclidified" distances (square root transformation)
    print("Performing PERMANOVA with euclidified distances...")
    euclidified_distances = np.sqrt(distance_matrix)
    euclidified_matrix_skbio = DistanceMatrix(euclidified_distances, ids=metadata.index)
    permanova_results_euclidified = permanova(
        euclidified_matrix_skbio,
        grouping=metadata['species'],
        permutations=999
    )
    
    r_squared_euclidified = calculate_r_squared(euclidified_distances, metadata['species'])
    
    # Save results
    results = {
        'Regular PERMANOVA': {
            'test statistic': permanova_results['test statistic'].item(),
            'p-value': permanova_results['p-value'].item(),
            'number of permutations': permanova_results['number of permutations'],
            'R-squared': r_squared
        },
        'Euclidified PERMANOVA': {
            'test statistic': permanova_results_euclidified['test statistic'].item(),
            'p-value': permanova_results_euclidified['p-value'].item(),
            'number of permutations': permanova_results_euclidified['number of permutations'],
            'R-squared': r_squared_euclidified
        }
    }
    
    # Save results to CSV
    results_df = pd.DataFrame(results).round(6)
    output_file = os.path.join(output_dir, 'permanova_results.csv')
    results_df.to_csv(output_file)
    print(f"\nResults saved to: {output_file}")
    
    # Also save metadata for reference
    metadata_file = os.path.join(output_dir, 'species_metadata.csv')
    metadata.to_csv(metadata_file)
    print(f"Species metadata saved to: {metadata_file}")
    
    # Print results
    print("\nPERMANOVA Results:")
    print("------------------")
    print("\nRegular PERMANOVA:")
    print(f"Test statistic: {results['Regular PERMANOVA']['test statistic']:.6f}")
    print(f"p-value: {results['Regular PERMANOVA']['p-value']:.6f}")
    print(f"R-squared: {results['Regular PERMANOVA']['R-squared']:.6f}")
    
    print("\nEuclidified PERMANOVA:")
    print(f"Test statistic: {results['Euclidified PERMANOVA']['test statistic']:.6f}")
    print(f"p-value: {results['Euclidified PERMANOVA']['p-value']:.6f}")
    print(f"R-squared: {results['Euclidified PERMANOVA']['R-squared']:.6f}")
    
    # Print some summary statistics
    print("\nSummary:")
    print(f"Number of samples: {len(metadata)}")
    print(f"Number of species: {metadata['species'].nunique()}")
    print("\nSamples per species:")
    print(metadata['species'].value_counts().describe())
    print("\nTop 10 most common species:")
    print(metadata['species'].value_counts().head(10))
    
    return results

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Perform PERMANOVA analysis on SimCLR embeddings')
    parser.add_argument('--embeddings_csv', type=str, required=True,
                      help='Path to CSV file containing embeddings')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    perform_simclr_permanova(args.embeddings_csv, args.output_dir)

if __name__ == "__main__":
    main()