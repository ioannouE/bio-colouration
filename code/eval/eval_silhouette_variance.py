import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, pairwise_distances
import os
import json
import argparse
from tqdm import tqdm

def calculate_silhouette_metrics(embeddings, labels, metric='euclidean'):
    """Calculate silhouette score directly using sklearn."""
    return float(silhouette_score(X=embeddings, labels=labels, metric=metric))

def extract_metadata_from_filename(filename):
    """
    Extract species and sex from filename, supporting multiple formats.
    Format 1: Genus_species_id_SEX... (e.g., 'Anthochaera_chrysoptera_1_M_12345')
    Format 2: SpeciesName_ID1_ID2 (e.g., 'Abraxas_fulvobasalis_1D_040917')
    """
    base_name = os.path.basename(filename)
    parts = base_name.split('_')

    # Try Format 1: Genus_species_id_SEX...
    if len(parts) >= 4 and parts[3] in ['M', 'F', 'U']:
        species = f"{parts[0]}_{parts[1]}"
        sex = parts[3]
        return species, sex

    # Try Format 2: SpeciesName_ID1_ID2
    if len(parts) > 2:
        species = "_".join(parts[:-2])
        # No reliable sex info in this format
        sex = None
        return species, sex

    # Fallback for unexpected formats
    return base_name, None

def calculate_variance_explained(embeddings, labels):
    """Calculate percentage of variance explained by labels."""

    overall_mean = embeddings.mean(axis=0)
    total_var = np.sum(((embeddings - overall_mean) ** 2).sum(axis=0))
    
    unique_labels = np.unique(labels)
    between_var = 0
    
    for label in unique_labels:
        mask = labels == label
        group_size = np.sum(mask)
        if group_size > 0:
            group_mean = embeddings[mask].mean(axis=0)
            between_var += group_size * np.sum((group_mean - overall_mean) ** 2)
    
    return float(between_var / total_var * 100) if total_var > 0 else 0.0

def filter_classes(labels, min_samples=2):
    """Filter out classes with fewer than min_samples samples."""
    unique_labels = np.unique(labels)
    valid_classes = []
    
    for label in unique_labels:
        if np.sum(labels == label) >= min_samples:
            valid_classes.append(label)
    
    mask = np.isin(labels, valid_classes)
    return mask, valid_classes

def analyze_embeddings(embeddings_csv, output_dir, num_samples=None, random_seed=42, calculate_r_squared=False):
    """Analyze SimCLR embeddings and calculate metrics."""
    
    # Load embeddings
    print("\nLoading embeddings...")
    df = pd.read_csv(embeddings_csv)
    embeddings = df.iloc[:, 1:].values  # First column is filename
    filenames = df.iloc[:, 0].values
    
    if num_samples is not None and num_samples < len(embeddings):
        print(f"\nSampling {num_samples} embeddings...")
        np.random.seed(random_seed)
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings = embeddings[indices]
        filenames = filenames[indices]
    else:
        print(f"\nUsing all {len(embeddings)} samples...")
    
    # Extract metadata
    print("\nExtracting metadata...")
    species_labels = []
    sex_labels = []
    class_labels = []
    
    for filename in tqdm(filenames):
        species, sex = extract_metadata_from_filename(filename)
        species_labels.append(species)
        sex_labels.append(sex if sex else 'U')
        class_labels.append(f"{species}_{sex}" if sex else species)
    
    species_labels = np.array(species_labels)
    sex_labels = np.array(sex_labels)
    class_labels = np.array(class_labels)
    
    # Filter classes for silhouette score
    min_samples_per_class = 2
    mask, valid_classes = filter_classes(class_labels, min_samples_per_class)
    
    embeddings_filtered = embeddings[mask]
    class_labels_filtered = class_labels[mask]
    species_labels_filtered = species_labels[mask]
    sex_labels_filtered = sex_labels[mask]
    
    normalized_embeddings = normalize(embeddings_filtered, norm='l2')

    # Print statistics
    print("\nDataset statistics:")
    print(f"Total samples: {len(embeddings)}")
    print(f"Total unique classes: {len(np.unique(class_labels))}")
    print(f"Classes with ≥{min_samples_per_class} samples: {len(valid_classes)}")
    print(f"Samples after filtering: {len(embeddings_filtered)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate silhouette scores
    print("\nCalculating Silhouette Scores...")
    silhouette_results = {}
    
    metrics = ['euclidean', 'cosine']
    for metric in metrics:
        print(f"\nCalculating {metric} silhouette scores...")
        # if metric == 'cosine':
        #     embeddings_to_use = normalize(embeddings_filtered, norm='l2', axis=1)
        # else:
        #     embeddings_to_use = (embeddings_filtered - embeddings_filtered.mean(axis=0)) / embeddings_filtered.std(axis=0)
            
        score = calculate_silhouette_metrics(normalized_embeddings, class_labels_filtered, metric=metric)
        silhouette_results[f'{metric}_distance'] = score
        print(f"Silhouette score ({metric}): {score:.4f}")
    
    # Calculate variance explained using all data
    print("\nCalculating variance explained...")
    variance_results = {
        'species_variance': calculate_variance_explained(embeddings, species_labels),
        'sex_variance': calculate_variance_explained(embeddings, sex_labels),
        'class_variance': calculate_variance_explained(embeddings, class_labels)
    }
    print(f"Variance explained by species: {variance_results['species_variance']:.1f}%")
    print(f"Variance explained by sex: {variance_results['sex_variance']:.1f}%")
    print(f"Variance explained by class: {variance_results['class_variance']:.1f}%")
    
    # Save results
    results = {
        'metadata': {
            'num_samples': len(embeddings),
            'num_species': len(np.unique(species_labels)),
            'num_sexes': len(np.unique(sex_labels)),
            'num_classes': len(np.unique(class_labels)),
            'num_valid_classes': len(valid_classes),
            'min_samples_per_class': min_samples_per_class,
            'samples_after_filtering': len(embeddings_filtered),
            'sex_labels_found': sorted(list(np.unique(sex_labels))),
            'embedding_dim': embeddings.shape[1],
            'random_seed': random_seed if num_samples is not None else None,
            'sampled_data': num_samples is not None
        },
        'silhouette_analysis': silhouette_results,
        'variance_explained': variance_results
    }
    
    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'embedding_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze SimCLR embeddings')
    parser.add_argument('--embeddings', help='Path to CSV file containing embeddings')
    parser.add_argument('--output', help='Directory to save results')
    parser.add_argument('--num-samples', type=int, help='Number of random samples to analyze. If not specified, uses all samples.')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--r-squared', action='store_true', help='Calculate R-squared score in addition to silhouette scores')
    
    args = parser.parse_args()
    analyze_embeddings(
        args.embeddings,
        args.output,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        calculate_r_squared=args.r_squared
    )