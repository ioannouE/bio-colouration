# Embedding Evaluation and Visualization

This module evaluates and visualizes the relationship between image embeddings (e.g., from ResNet, SimCLR) or transformed embeddings (e.g., PCA) and semantic features such as color, shape, or class labels.

----------

## Features

### 1. **Evaluation of Embeddings**

-   Supports both **raw embeddings** and **PCA-transformed embeddings**
    
-   Computes relationships between embeddings and categorical features using:
    
    -   **Classification accuracy**: e.g., k-NN, linear probing (in `eval_utils`)
        
    -   **Statistical scores**:
        
        -   `R²` (from ANOVA-style metrics)
            
        -   **Kruskal–Wallis H test**
            
        -   Spearman correlation (optionally)

### 2. **Visualization**

-   Creates **scatter plots with image thumbnails** for selected embedding dimensions
    
-   Two types of embedding visualizations:
    
    -   **Raw embedding space** (e.g., x1–x1024)
        
    -   **PCA space** (e.g., PC1–PC50)

Plots are sorted based on the top n (default is 6) dimensions with highest `anova_r2` scores per category.

----------

## Input Requirements

-   A `.csv` file containing:
    
    -   Image filenames
        
    -   Embedding columns (e.g., x1, x2, ..., x1024)
        
    -   Metadata labels (e.g., color, shape)
        
-   Corresponding images in a directory
    

----------

## Usage

```bash
python evaluate.py \
  --csv_path path/to/your_data.csv \
  --image_dir path/to/images/ \
  --output_dir path/to/save/plots/ \
  --labels color shape \
  --emb_num 1024 \
  --emb_prefix x \
  --pca_nc 50 \
  --plot_nc 6 \
  --subset_size 200

```

----------

## Outputs

For each label (e.g., color, shape), the script will generate:

-   Top PCA visualizations (`*_pca_0_v_1.png`, ...)
    
-   Top embedding visualizations (`*_emb_0_v_1.png`, ...)
    
-   Statistical analysis results (e.g., ANOVA `R²`, Kruskal scores)
    

----------

## Dependencies

-   `pandas`
    
-   `scikit-learn`
    
-   `scipy`
    
-   Custom modules:
    
    -   `utils.scatter_with_thumbnails` from the colour features, make sure to adjust `sys.path` to locate these modules if running from another environment.
        
    -   `eval_utils.analyze_embedding_df`
        
    -   `eval_utils.run_pca_with_variance_columns`
        



