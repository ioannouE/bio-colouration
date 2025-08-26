import os, sys
import argparse
## utils from the colour feature
# sys.path.append("C:/Users/Yichen/OneDrive/work/codes/shef_bird_colour/image_features")
import utils

import eval_utils
import pandas as pd
import json
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_plots(df, df_stat, plot_nc, values, label, args, output_dir=None):
    """Plot multiple scatter-with-thumbnail plots in one figure"""
    
    n_plots = plot_nc // 2
    n_cols = 2  # number of columns of subplots
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols, 10 * n_rows), dpi=150)
    axes = axes.flatten()  # make it easy to index

    for plot_idx in range(n_plots):
        i = plot_idx * 2
        x_col = df_stat.iloc[i]['dim']
        y_col = df_stat.iloc[i + 1]['dim']
        x_r2 = df_stat.iloc[i]['anova_r2']
        y_r2 = df_stat.iloc[i + 1]['anova_r2']
        
        x_label = f"{x_col} (R²: {x_r2:.4f})"
        y_label = f"{y_col} (R²: {y_r2:.4f})"

        if "PC" in x_col:
            x_pc_var = df_stat.iloc[i]['pc_var']
            y_pc_var = df_stat.iloc[i+1]['pc_var']
            title = f"{label} {x_col} (PC var {x_pc_var*100:.2f}) vs {y_col} (PC var {y_pc_var*100:.2f})"
        else:
            title = f"{label} {x_col} vs {y_col}"

        print(f"Plotting subplot {plot_idx+1}: {x_col} vs {y_col}")
        
        df_plot = df.copy()
        if args.subset_size is not None and args.subset_size > 0:
            df_plot = df_plot.sample(n=args.subset_size, random_state=42)

        utils.scatter_with_thumbnails(
            df=df_plot,
            x_col=x_col,
            y_col=y_col,
            image_dir=args.image_dir,
            image_col=args.image_col,
            mask_col=args.mask_col,
            mask_dir=args.mask_dir,
            thumb_size=(40, 40),
            ax=axes[plot_idx],  # use the corresponding subplot axis
            normalize=False,
            skewed_fix=False,
            title=title,
            x_label=x_label,
            y_label=y_label,
            plot_show = False
        )

    # Remove unused axes (if any)
    # for j in range(n_plots, len(axes)):
    #     fig.delaxes(axes[j])
    if output_dir is None:
        output_path = os.path.join(args.output_dir, f"{label}_{values}_combined.png")
    else:
        output_path = os.path.join(output_dir, f"{label}_{values}_combined.png")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Combined plot saved to: {output_path}")



def plot_top_dim_acc(df, df_stat, plot_nc, values, labels, args, fig_title, type = "pca", output_dir=None):
    dims = df_stat.dim.values
    
    if type == "pca":
        pc_vars = df_stat.pc_var.values
    
    n_cols = plot_nc
    n_rows = math.ceil(plot_nc / n_cols)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), dpi=150)
    axes = axes.flatten()
    
    # Storage for all results
    all_results = []

    for i in range(1, plot_nc+1):
        sub_dims = dims[:i]
        if type == "pca":
            sum_pc_vars = sum(pc_vars[:i])
            sub_title = f"Top {i} PCA dims: {sub_dims} (PC var: {sum_pc_vars*100:.2f})"
        else:
            sub_title = f"Top {i} dims: {sub_dims}"
        
        df_dim_cls_result = eval_utils.evaluate_embedding_classification(df=df,
                                                                     embed_cols=sub_dims, 
                                                                     category_cols=labels)

        # Add metadata columns for stacking
        df_dim_cls_result["type"] = type
        df_dim_cls_result["top_n"] = i
        df_dim_cls_result["top_n_dims"] = ", ".join(sub_dims)  # make it string for saving

        all_results.append(df_dim_cls_result)


        eval_utils.plot_accuracy_by_method_and_category(df_dim_cls_result,
                                                        title=sub_title, 
                                                        figsize=(10, 6), 
                                                        ax = axes[i - 1],
                                                        plot_show=False)

    if output_dir is None:
        output_path = os.path.join(args.output_dir, fig_title)
        output_path_fig_top_n = os.path.join(args.output_dir, os.path.splitext(fig_title)[0] + "_by_top_n.png")
        csv_output_path = os.path.join(args.output_dir, os.path.splitext(fig_title)[0] + ".csv")
    else:
        output_path = os.path.join(output_dir, fig_title)
        output_path_fig_top_n = os.path.join(output_dir, os.path.splitext(fig_title)[0] + "_by_top_n.png")
        csv_output_path = os.path.join(output_dir, os.path.splitext(fig_title)[0] + ".csv")
    
    # Save combined figure
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Combined subplot saved to: {output_path}")





    # Save stacked DataFrame
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(csv_output_path, index=False)
    print(f"[INFO] Stacked classification results saved to: {csv_output_path}")
    
        ### Plot accuracy by top_n for each method and category
    eval_utils.plot_accuracy_by_top_n(df_all, 
                                    output_path=output_path_fig_top_n)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process UMAP embeddings.')
    parser.add_argument('--csv_path', type=str, required=True, help='csv of the embeddings and labels')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the results')
    parser.add_argument('--labels', nargs="+", help='list of strings of the labels')
    
    parser.add_argument('--image_col', type=str, default = 'filename', help='the column name of the image file names')
    parser.add_argument('--mask_col', type=str, default=None, help='The column name of the mask file names (optional)')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory containing masks (optional)')
    
    parser.add_argument('--emb_num', type=int, default = 1024, help='number of embeddings to use')
    parser.add_argument('--emb_prefix', type=str, default='x',  help='prefix of embeddings to use')
    parser.add_argument('--pca_nc', type=int, default=50,  help='number of components for PCA')
    parser.add_argument("--plot_nc", type=int, default=8,
                    help="Number of top components to visualize (must be even)")
    parser.add_argument('--subset_size', type=int, default=200, help='Size of the random subset to use')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    output_structure = {
        "embeddings": "transformed_embeddings",
        "classification": "classification",
        "similarity": "similarity_visuals",
        "scatter": "scatter_thumbnails"
    }

    output_paths = {k: os.path.join(args.output_dir, v) for k, v in output_structure.items()}
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # Print args
    print("\nParsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

      
    plot_nc = args.plot_nc
    assert plot_nc % 2 == 0, "plot_nc must be even"

    ## finishing setup the output folder
    
    df_emb = pd.read_csv(args.csv_path)

    embed_cols = [f'{args.emb_prefix}{i}' for i in range(1, args.emb_num+1)]
    labels = args.labels

    df_embed_stats = eval_utils.analyze_embedding_df(
        df=df_emb,
        embed_cols=embed_cols,
        categorical_cols=labels,

    )
    



    df_pca, var_explained = eval_utils.run_pca_with_variance_columns(df=df_emb,
                                                                          embed_cols=embed_cols,
                                                                          n_components=args.pca_nc)
    # df_pca: shape: (n_samples, labels + n_embed_cols + n_pca)
    
    pca_cols = [f'PC{i}' for i in range(1, args.pca_nc+ 1)]
    df_pca_stats = eval_utils.analyze_embedding_df(
        df=df_pca,
        embed_cols=pca_cols,
        categorical_cols=labels,
        is_pca=True
    )
    
    ### Plot embedding similarity grid using distance df using all embeddings ###
    eval_utils.plot_embedding_similarity_grid(
        df=df_emb,
        embed_cols=embed_cols,
        image_dir=args.image_dir,
        output_path=os.path.join(output_paths['similarity'],'dist_similar.png'),
        n_samples=5,
        top_n=5,
        mask_col=args.mask_col,
        mask_dir=args.mask_dir
    )
    
    #### Plot scatter with Thumbnails ####
    for label in labels:
        df_temp = df_pca_stats.loc[df_pca_stats['category'] == label, :]
        df_temp.sort_values('anova_r2', ascending=False , inplace=True)
        
        plot_plots(df = df_pca, df_stat = df_temp,
                   plot_nc = plot_nc, values = "pca",
                   label = label, 
                   args = args,
                   output_dir = output_paths['scatter'])
        
        # new_labels = ["shape" , "color1" , "filename"]
        
        ## Iterate over number of pca components and see how the accuracy pcs can predict labels
        plot_top_dim_acc(df = df_pca, df_stat = df_temp, plot_nc = plot_nc,
                        values = None, labels = labels, args = args, 
                        fig_title = f'classification_top_{plot_nc}_pc_on_{label}.png' ,
                        type="pca",
                        output_dir=output_paths['classification'])
    
        dims = df_temp.dim.values
        for i in range(1, plot_nc+1):
            sub_dims = dims[:i]
            eval_utils.plot_embedding_similarity_grid(
                df=df_pca,
                embed_cols=sub_dims,
                image_dir=args.image_dir,
                output_path=os.path.join(output_paths['similarity'],f'dist_similar_top_{i}_pca_from_{label}.png'),
                n_samples=5,
                top_n=5,
                mask_col=args.mask_col,
                mask_dir=args.mask_dir
                
            )
   
    
    
        ############## Embedding ##############
        ## Iterate over number of pca components and see how the accuracy pcs can predict labels
        df_temp = df_embed_stats.loc[df_embed_stats['category'] == label, :]
        df_temp.sort_values('anova_r2', ascending=False , inplace=True)
        
        plot_plots(df = df_emb, df_stat = df_temp,
                   plot_nc = plot_nc, values = "emb",
                   label = label, 
                   args = args,
                   output_dir = output_paths['scatter'])
    
    

        plot_top_dim_acc(df = df_emb, df_stat = df_temp, plot_nc = plot_nc,
                    values = None, labels = labels, args = args, 
                    fig_title = f'classification_top_{plot_nc}_emb_on_{label}.png',
                    type="emb",
                    output_dir=output_paths['classification'])

        dims = df_temp.dim.values
        for i in range(1, plot_nc+1):
            sub_dims = dims[:i]
            eval_utils.plot_embedding_similarity_grid(
                df=df_emb,
                embed_cols=sub_dims,
                image_dir=args.image_dir,
                output_path=os.path.join(output_paths['similarity'],f'dist_similar_top_{i}_emb_from_{label}.png'),
                n_samples=5,
                top_n=5,
                mask_col=args.mask_col,
                mask_dir=args.mask_dir
            )

    
    #### Plot PCA by variance explained ####
    plot_plots(df = df_pca, df_stat =  df_pca_stats.drop_duplicates(subset=['dim']),
                plot_nc = 16, values = "pca",
                label = "",args = args,
                output_dir = output_paths['scatter'])


    #### Writing CSV ####
    df_cls_result = eval_utils.evaluate_embedding_classification(df=df_emb, embed_cols=embed_cols, category_cols=labels)
    ### Plot accuracy by method and category
    eval_utils.plot_accuracy_by_method_and_category(df_cls_result, title="Accuracy by Method and Category", 
                                                    figsize=(10, 6), 
                                                    output_path=os.path.join(output_paths['classification'],'classification_results_by_emb.png'))
    df_cls_result.to_csv(os.path.join(output_paths['classification'], 'classification_results_by_emb.csv'), index=False)
    

    df_pca.to_csv(os.path.join(output_paths['embeddings'], 'pca.csv'), index=False)
    df_embed_stats.to_csv(os.path.join(output_paths['embeddings'], 'embed_stats.csv'), index=False)
    df_pca_stats.to_csv(os.path.join(output_paths['embeddings'], 'pca_stats.csv'), index=False)
    
    


    
    
    