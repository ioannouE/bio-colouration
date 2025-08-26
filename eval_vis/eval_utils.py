

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, homogeneity_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

import os
import cv2


ROUND_DIGIT =4
def get_dist_df(df, query_idx, embed_cols, metric='euclidean', round_digits=4):
    """
    Given a DataFrame with embeddings and metadata, return a DataFrame of all samples 
    with computed distances and similarity scores to the query, sorted by distance.

    Args:
        df: DataFrame containing embedding columns and metadata
        query_idx: index (or row number) of the query sample in df
        embed_cols: list of column names for embedding dimensions
        metric: 'euclidean' or 'cosine'
        round_digits: number of decimal places to round distances

    Returns:
        df_with_score: DataFrame of all other samples with columns: 'distance', 'similarity', sorted by distance
    """
    embeddings = df[embed_cols].values
    query = embeddings[query_idx].reshape(1, -1)

    if metric == 'euclidean':
        dists = cdist(query, embeddings, metric='euclidean')[0]
        similarity = -dists  # lower distance = more similar
    elif metric == 'cosine':
        sims = cosine_similarity(query, embeddings)[0]
        dists = 1 - sims  # convert similarity to distance
        similarity = sims  # higher similarity = more similar
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    df_with_score = df.copy()
    df_with_score['similarity'] = similarity
    df_with_score['distance'] = np.round(dists, round_digits)

    # Exclude the query itself
    df_with_score = df_with_score.drop(index=query_idx)

    return df_with_score.sort_values('distance', ascending=True)



from PIL import Image

# def plot_ranked_images(df_ranked, N=5, query_img_path=None, image_col='filename',
#                        image_dir='.', figsize=(15, 6), output_path=None, title_prefix='',
#                        max_width=300 , plot_show=True):
#     """
#     Plot top-N closest and bottom-N furthest images in two rows, optionally with a query image on the far left.

#     Args:
#         df_ranked: DataFrame sorted by distance, with columns including filename and distance
#         N: Number of top and bottom images to display
#         query_img_path: Optional path to query image (shown on far left)
#         image_col: Column in df_ranked with image filenames
#         image_dir: Directory containing the image files
#         figsize: Size of the figure
#         output_path: If given, saves the figure to this path
#         title_prefix: Text prefix for image titles (e.g., "Top", "Bottom")
#         max_width: Maximum width in pixels for displaying any image (preserve aspect ratio)
#     """
#     top_df = df_ranked.head(N)
#     bottom_df = df_ranked.tail(N)
#     total_cols = N + (1 if query_img_path else 0)

#     fig, axes = plt.subplots(2, total_cols, figsize=figsize)

#     # Load and display query image
#     if query_img_path:
#         ax_top = axes[0, 0]
#         ax_bottom = axes[1, 0]
#         try:
#             img = Image.open(query_img_path)
#             if img.width > max_width:
#                 new_height = int(max_width * img.height / img.width)
#                 img = img.resize((max_width, new_height))
#             ax_top.imshow(img)
#             ax_top.set_title("Query")
#         except:
#             ax_top.text(0.5, 0.5, 'Query not found', ha='center')
#         ax_top.axis('off')
#         ax_bottom.axis('off')

#     offset = 1 if query_img_path else 0

#     for i, (_, row) in enumerate(top_df.iterrows()):
#         ax = axes[0, i + offset]
#         img_path = os.path.join(image_dir, row[image_col])
#         try:
#             img = Image.open(img_path)
#             if img.width > max_width:
#                 new_height = int(max_width * img.height / img.width)
#                 img = img.resize((max_width, new_height))
#             ax.imshow(img)
#             ax.set_title(f"Top {i+1}\nDist: {row['distance']}")
#         except:
#             ax.text(0.5, 0.5, 'Image not found', ha='center')
#         ax.axis('off')

#     for i, (_, row) in enumerate(bottom_df.iterrows()):
#         ax = axes[1, i + offset]
#         img_path = os.path.join(image_dir, row[image_col])
#         try:
#             img = Image.open(img_path)
#             if img.width > max_width:
#                 new_height = int(max_width * img.height / img.width)
#                 img = img.resize((max_width, new_height))
#             ax.imshow(img)
#             ax.set_title(f"Bottom {i+1}\nDist: {row['distance']}")
#         except:
#             ax.text(0.5, 0.5, 'Image not found', ha='center')
#         ax.axis('off')

#     plt.tight_layout()
#     if output_path:
#         plt.savefig(output_path)
#         plt.close(fig)
#     elif plot_show:
#         plt.show()

def plot_ranked_images(axs, df_ranked, N=5, query_img_path=None, 
                       image_col='filename',
                       image_dir='.', 
                        mask_dir=None,
                        mask_col=None,
                       title_prefix='', max_width=300):
    """
    Plot top-N and bottom-N ranked images into provided subplot axes.

    Args:
        axs: 2-row list or array of matplotlib axes [ [top_row_axes], [bottom_row_axes] ]
        df_ranked: DataFrame sorted by distance
        N: number of top/bottom images to plot
        query_img_path: path to query image
        image_col: DataFrame column with image filenames
        image_dir: path to folder with images
        title_prefix: text prefix for image title (e.g. Euclidean)
        max_width: max image width (preserving aspect ratio)
    """
    
    def apply_mask_to_image(img, mask_path):
        if os.path.isfile(mask_path):
            try:
                mask = cv2.imread(mask_path, 0)
                mask = (mask > 0).astype(np.uint8)

                # Apply bounding box crop
                ys, xs = np.where(mask)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    img = img[y_min:y_max+1, x_min:x_max+1]
                    mask = mask[y_min:y_max+1, x_min:x_max+1]

                    # Optional: apply mask to crop transparency (if needed)
                    # img = cv2.bitwise_and(img, img, mask=mask)
                    # Create white background
                    white_bg = np.ones_like(img, dtype=np.uint8) * 255

                    # Copy foreground where mask > 0
                    for c in range(3):
                        white_bg[:, :, c] = np.where(mask > 0, img[:, :, c], 255)

                    img = white_bg
                    return img
                    
            except Exception as e:
                print(f"Failed to process mask for {img_path}: {e}")   
                
    top_df = df_ranked.head(N)
    bottom_df = df_ranked.tail(N)

    if query_img_path:
        try:
            img = Image.open(query_img_path)
            
            if img.width > max_width:
                new_height = int(max_width * img.height / img.width)
                img = img.resize((max_width, new_height))
            axs[0][0].imshow(img)
            axs[0][0].set_title("Query")
        except:
            axs[0][0].text(0.5, 0.5, 'Query not found', ha='center')
        axs[0][0].axis('off')
        axs[1][0].axis('off')

    offset = 1 if query_img_path else 0

    for i, (_, row) in enumerate(top_df.iterrows()):
        ax = axs[0][i + offset]
        img_path = os.path.join(image_dir, row[image_col])
        try:
            img = Image.open(img_path)
            #### Load and apply mask to img if provided ###
            if mask_dir is not None and mask_col in row:
                mask_path = os.path.join(mask_dir, row[mask_col])
                img = apply_mask_to_image(np.array(img), mask_path)
                img = Image.fromarray(img)
            
            
            if img.width > max_width:
                new_height = int(max_width * img.height / img.width)
                img = img.resize((max_width, new_height))
            ax.imshow(img)
            ax.set_title(f"{title_prefix}Top {i+1}\nDist: {row['distance']}\n{row[image_col]}")
        except:
            ax.text(0.5, 0.5, 'Image not found', ha='center')
        ax.axis('off')

    for i, (_, row) in enumerate(bottom_df.iterrows()):
        ax = axs[1][i + offset]
        img_path = os.path.join(image_dir, row[image_col])
        try:
            img = Image.open(img_path)

            #### Load and apply mask to img if provided ###
            if mask_dir is not None and mask_col in row:
                mask_path = os.path.join(mask_dir, row[mask_col])
                img = apply_mask_to_image(np.array(img), mask_path)
                img = Image.fromarray(img)            

            if img.width > max_width:
                new_height = int(max_width * img.height / img.width)
                img = img.resize((max_width, new_height))
            ax.imshow(img)
            ax.set_title(f"{title_prefix}Bottom {i+1}\nDist: {row['distance']}\n{row[image_col]}")
        except:
            ax.text(0.5, 0.5, 'Image not found', ha='center')
        ax.axis('off')






def plot_embedding_similarity_grid(df, embed_cols, image_dir, output_path, image_col='filename',
                                    mask_dir=None,mask_col=None,
                                   n_samples=5, top_n=5, seed=0, max_width=300, figsize_per_plot=(15, 6)):
    """
    Create a grid of subplots showing similarity results (top-N and bottom-N) for n_samples and 2 metrics.
    
    Basically does the codes below
        df_dist_eucl = eval_utils.get_dist_df(df=df_emb,
                                                    query_idx=42, embed_cols=embed_cols, metric='euclidean')

        eval_utils.plot_ranked_images(
            df_ranked=df_dist_eucl,  
            image_dir=args.image_dir,
            title_prefix='Top ',
            output_path= os.path.join(args.output_dir,'euclidean_top_n.png')     )
        
    Args:
        df: DataFrame containing embeddings and metadata
        embed_cols: list of embedding column names
        image_dir: folder containing image files
        output_path: path to save the final figure
        image_col: column with image filenames
        n_samples: number of random samples to evaluate
        top_n: number of top/bottom neighbors to show
        seed: random seed for reproducibility
        max_width: max image width (pixels)
        figsize_per_plot: tuple indicating size of one subplot (width, height)
    """
    np.random.seed(seed)
    query_indices = np.random.choice(len(df), n_samples, replace=False)

    metrics = ['euclidean', 'cosine']
    n_metrics = len(metrics)
    n_cols = (top_n + 1)* n_metrics  # +1 for the query image
    total_rows = n_samples 

    fig, axes = plt.subplots(
        nrows=total_rows * 2,  # 2 rows per subplot (top/bottom)
        ncols=n_cols,
        figsize=(figsize_per_plot[0] * n_metrics, figsize_per_plot[1] * n_samples)
    )

    if isinstance(axes, np.ndarray):
        axes = axes.reshape(total_rows* 2, n_cols)

    

    for i, query_idx in enumerate(query_indices):
        for j, metric in enumerate(metrics):
            # row_base = (i * n_metrics + j) * 2
            col_offset = j * (top_n + 1)
            df_dist = get_dist_df(df, query_idx, embed_cols, metric=metric)
            query_img_path = os.path.join(image_dir, df.loc[query_idx, image_col])
            axs_pair = [
                axes[i * 2, col_offset: col_offset + top_n + 1],
                axes[i * 2 + 1, col_offset: col_offset + top_n + 1]
            ]


            plot_ranked_images(
                axs=axs_pair,
                df_ranked=df_dist,
                N=top_n,
                query_img_path=query_img_path,
                image_col=image_col,
                image_dir=image_dir,
                title_prefix=f'{metric.title()} ',
                max_width=max_width,
                mask_dir = mask_dir,
                mask_col= mask_col
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)



##### evaluate_embedding_classification ######

def knn_evaluate(embeddings, labels, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    acc = accuracy_score(labels, preds)
    return acc

def centroid_classify(embeddings, labels):
    unique_labels = np.unique(labels)
    centroids = {label: embeddings[labels == label].mean(axis=0) for label in unique_labels}
    
    preds = []
    for emb in embeddings:
        pred = min(centroids, key=lambda c: np.linalg.norm(emb - centroids[c]))
        preds.append(pred)
    
    acc = accuracy_score(labels, preds)
    return acc


def linear_probe(embeddings, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc




def clustering_purity(embeddings, labels, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return homogeneity_score(labels, km.labels_)




def evaluate_embedding_classification(df, embed_cols, category_cols, k=5, test_size=0.2):
    """
    Evaluate embedding quality on multiple categorical labels using different classifiers.

    Args:
        df: DataFrame containing embedding columns and category columns
        embed_cols: list of embedding column names (e.g., x1 to x1024)
        category_cols: list of categorical column names to evaluate (e.g., ['shape', 'colour'])
        k: number of neighbors for k-NN
        test_size: test size for linear probing

    Returns:
        results: DataFrame with accuracy scores for each method and category
    """
    results = []

    embeddings = df[embed_cols].values

    for category in category_cols:
        labels = df[category].values

        try:
            X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, stratify=labels)
        except:
            print(f"[ERROR] Stratified split failed for category {category}. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size)
        
        # k-NN
        # print("Evaluating k-NN...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn_preds = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_preds)

        # Nearest Class Centroid
        # print("Evaluating Nearest Class Centroid...")
        centroids = {label: embeddings[labels == label].mean(axis=0) for label in np.unique(labels)}
        ncc_preds = [min(centroids, key=lambda c: np.linalg.norm(e - centroids[c])) for e in embeddings]
        ncc_acc = accuracy_score(labels, ncc_preds)

        # Linear Probing
        # print("Evaluating Linear Probe...")
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        lp_preds = clf.predict(X_test)
        lp_acc = accuracy_score(y_test, lp_preds)

        # Clustering Homogeneity
        # print("Evaluating Clustering Homogeneity...")
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings)
        homogeneity = homogeneity_score(labels, cluster_labels)

        results.append({
            'category': category,
            'method': 'kNN',
            'accuracy': round(knn_acc , ROUND_DIGIT)
        })
        results.append({
            'category': category,
            'method': 'NearestClassCentroid',
            'accuracy': round(ncc_acc, ROUND_DIGIT)
        })
        results.append({
            'category': category,
            'method': 'LinearProbe',
            'accuracy': round(lp_acc, ROUND_DIGIT)
        })
        results.append({
            'category': category,
            'method': 'ClusteringHomogeneity',
            'accuracy': round(homogeneity, ROUND_DIGIT)
        })

    return pd.DataFrame(results)

def plot_accuracy_by_method_and_category(df, 
                                         title="Accuracy by Method and Category",
                                         figsize=(10, 6), 
                                         ax = None,
                                         output_path=None,
                                         plot_show = True):
    """
    Plot accuracy by method and category using matplotlib only.

    Args:
        df (pd.DataFrame): must contain columns ['category', 'method', 'accuracy']
        title (str): plot title
        figsize (tuple): figure size
        output_path (str or None): path to save figure if provided
    """
    categories = sorted(df['category'].unique())
    methods = sorted(df['method'].unique())

    # Bar settings
    bar_width = 0.8 / len(methods)
    x = np.arange(len(categories))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        heights = [method_data[method_data['category'] == cat]['accuracy'].values[0] if not method_data[method_data['category'] == cat].empty else 0 for cat in categories]
        ax.bar(x + i * bar_width, heights, width=bar_width, label=method)

    # Formatting
    ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(title="Method")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300)
            print(f"[INFO] Plot saved to {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save plot to {output_path}: {e}")
    elif plot_show:
        plt.show()

def plot_accuracy_by_top_n(df, output_path=None):
    """
    Plot accuracy vs top_n for each (category, method) pair in the dataframe.

    Args:
        df (pd.DataFrame): must contain 'category', 'method', 'accuracy', 'top_n'
        output_path (str or None): path to save the figure if provided
    """
    import matplotlib.pyplot as plt

    categories = df['category'].unique()
    methods = df['method'].unique()

    fig, axes = plt.subplots(len(categories), len(methods), figsize=(14, 6), sharex=True, sharey=True)

    for i, category in enumerate(categories):
        for j, method in enumerate(methods):
            ax = axes[i, j] if len(categories) > 1 else axes[j]
            sub_df = df[(df['category'] == category) & (df['method'] == method)]
            ax.plot(sub_df['top_n'], sub_df['accuracy'], marker='o')
            ax.set_title(f"{category} - {method}")
            ax.set_ylim(0, 1.1)
            
            xticks = sorted(sub_df['top_n'].unique())
            ax.set_xticks(xticks)
        
            if i == len(categories) - 1:
                ax.set_xlabel("top_n")
            if j == 0:
                ax.set_ylabel("accuracy")

    plt.suptitle("Accuracy vs top_n by Category and Method")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        try:
            plt.savefig(output_path, dpi=300)
            print(f"[INFO] Plot saved to {output_path}")
        except Exception as e:
            print(f"[ERROR] Could not save plot: {e}")
    else:
        plt.show()


################### Calculate the stat vs embedding ##############



from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def analyze_embedding_df(df, embed_cols, numeric_cols=[], categorical_cols=[], is_pca=False):
    """
    Args:
        df: pandas DataFrame with embedding columns and metadata
        embed_cols: list of embedding column names (e.g., ['x1', ..., 'x1024'])
        numeric_cols: list of continuous property column names
        categorical_cols: list of categorical property column names
        method: 'pearson' / 'spearman' for numeric; 'f_oneway' / 'kruskal' for categorical
    Returns:
        DataFrame with scores and p-values
    """
    results = []

    for dim_col in embed_cols:
        values = df[dim_col].values

        # Numeric properties: correlation
        for prop in numeric_cols:
            prop_vals = df[prop].values
            # if method == 'spearman':
            #     corr, pval = spearmanr(values, prop_vals)

            corr, pval = pearsonr(values, prop_vals)
            results.append({
                'dim': dim_col,
                'category': prop,
                'type': 'numeric',
                'pearsonr': corr,
                'pearson_p': pval
            })

        # Categorical properties: ANOVA or Kruskal-Wallis
        for prop in categorical_cols:
            groups = [values[df[prop] == cat] for cat in df[prop].dropna().unique()]
            
            kruskal_stat, kruskal_p = kruskal(*groups)
            
            f_stat, f_p = f_oneway(*groups)
                
            # Calculate R² for ANOVA
            # This is a simplified version of the ANOVA R² calculation
            groups = [values[df[prop] == cat] for cat in df[prop].dropna().unique()]

            # Total sum of squares
            grand_mean = np.mean(values)
            ss_total = np.sum((values - grand_mean)**2)

            # Between-group sum of squares
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

            r2 = ss_between / ss_total if ss_total > 0 else 0

            if is_pca:
                results.append({
                    'dim': dim_col,
                    'category': prop,
                    'type': 'categorical',
                    'kruskal': round(kruskal_stat, ROUND_DIGIT),
                    'kruskal_p': round(kruskal_p, ROUND_DIGIT),
                    'f_oneway': round(f_stat, ROUND_DIGIT),
                    "f_oneway_p": round(f_p, ROUND_DIGIT),
                    'anova_r2': round(r2, ROUND_DIGIT),
                    'pc_var' :  round(df[dim_col+"_var"].values[0], ROUND_DIGIT)
                })# For PCA, we might want to use a different method for categorical properties
                # For now, we will keep the same approach as above
                pass
            else:
            
                results.append({
                    'dim': dim_col,
                    'category': prop,
                    'type': 'categorical',
                    'kruskal': round(kruskal_stat, ROUND_DIGIT),
                    'kruskal_p': round(kruskal_p, ROUND_DIGIT),
                    'f_oneway': round(f_stat, ROUND_DIGIT),
                    "f_oneway_p": round(f_p, ROUND_DIGIT),
                    'anova_r2': round(r2, ROUND_DIGIT)
                })

    return pd.DataFrame(results)


def compute_anova_r2(df, embed_cols, category_col):
    """
    Returns R² for how much variance in each embedding dim is explained by the category_col
    """
    results = []

    for dim in embed_cols:
        y = df[dim].values
        groups = [y[df[category_col] == cat] for cat in df[category_col].dropna().unique()]

        # Total sum of squares
        grand_mean = np.mean(y)
        ss_total = np.sum((y - grand_mean)**2)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

        r2 = ss_between / ss_total if ss_total > 0 else 0
        results.append({
            'dim': dim,
            'category': category_col,
            'r2': r2
        })

    return pd.DataFrame(results)


def run_pca_with_variance_columns(df, embed_cols, n_components=50):
    """
    Performs PCA on embedding columns and returns a DataFrame containing:
    - PC1, PC2, ..., PCn
    - PC1_var, PC2_var, ..., PCn_var (repeating the explained variance for each sample)
    
    Returns:
        df_with_pca: original df + PC columns + PC variance columns
        explained_variance: list of explained variance per PC
    """
    pca = PCA(n_components=n_components)
    X = df[embed_cols].values
    X_pca = pca.fit_transform(X)

    # PC column names
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    var_cols = [f'{col}_var' for col in pc_cols]

    # PCA values DataFrame
    pca_df = pd.DataFrame(X_pca, columns=pc_cols, index=df.index)

    # Variance explained
    var_explained = pca.explained_variance_ratio_

    # Repeat each variance across rows
    var_df = pd.DataFrame(
        np.tile(var_explained, (len(df), 1)), 
        columns=var_cols, 
        index=df.index
    )

    # Combine everything
    df_with_pca = pd.concat([df, pca_df, var_df], axis=1)
    return df_with_pca, var_explained


def print_top_pca_variance(var_explained, top_n=10):
    print(f"Top {top_n} Principal Components Variance Explained:")
    for i in range(top_n):
        print(f"  PC{i+1}: {var_explained[i]:.4f} ({var_explained[i]*100:.2f}%)")
        
        


# def plot_plots(df, df_stat, plot_nc, values, label, args):
#     """Plot images with thumbnails"""
#     for i in range(0, plot_nc, 2):
#         x_col = df_stat.iloc[i]['dim']
#         y_col = df_stat.iloc[i + 1]['dim']
        
#         x_r2 = df_stat.iloc[i]['anova_r2']
#         y_r2 = df_stat.iloc[i + 1]['anova_r2']
        
#         print(f"plotting {label} with {x_col} and {y_col}")
#         print(f"x anova_r2 of {x_col}: {x_r2:.4f} and y anova_r2 of {y_col}: {y_r2:.4f}")
        
#         x_label = f"{x_col} (R²: {x_r2:.4f})"
#         y_label = f"{y_col} (R²: {y_r2:.4f})"
#         if "PC" in x_col:
#             x_pc_var = df_stat.iloc[i]['pc_var']
#             y_pc_var = df_stat.iloc[i+1]['pc_var'] 
#             title = f"{label} {x_col} (PC var {x_pc_var*100:.2f}) vs {y_col} (PC var {y_pc_var*100:.2f})"
#         else:
#             title = f"{label} {x_col} vs {y_col}"
        
#         fig_path = os.path.join(args.output_dir, f'{label}_{values}_{i}_v_{i+1}.png')

#         if args.subset_size is not None or args.subset_size > 0:
#             df = df.sample(n=args.subset_size, random_state=42)
        
#         utils.scatter_with_thumbnails(
#             df=df,
#             x_col=x_col,
#             y_col=y_col,
#             image_dir=args.image_dir,
#             image_col='filename',
#             mask_dir=None,
#             thumb_size=(40, 40),
#             figsize=(15, 10),
#             dpi=150,
#             output_path=fig_path,
#             normalize=False,
#             skewed_fix=False,
#             title = title,
#             x_label=x_label,
#             y_label=y_label
#         )        
