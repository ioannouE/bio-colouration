import warnings
warnings.filterwarnings("ignore")

import os
import torch
import yaml
import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl
# from simclr_birdcolour import SimCLRModel, SimCLRDataModule
# from simclr_birdcolour_tune_eval import perform_permanova, generate_embeddings, plot_knn_examples
# from simclr_birdcolour_tune_eval import plot_knn_examples
import datetime as dt
import re
import csv
import matplotlib.pyplot as plt
import json
from simclr_birdcolour_kornia02 import SimCLRModel, SimCLRDataModule, plot_knn_examples


def generate_embeddings_from_layer(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model, extracting features after backbone layer 3.
    """

    # Create a new model consisting of the first 4 layers of the backbone
    # and an adaptive average pooling layer
    feature_extractor = torch.nn.Sequential(
        *list(model.backbone.children())[:7],  # Layers 0, 1, 2, 3
        torch.nn.AdaptiveAvgPool2d((1, 1)) # Add pooling
    ).to(model.device) # Move the new module to the same device as the model
    feature_extractor.eval() # Set the feature extractor to evaluation mode

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            # Pass image through the truncated backbone and pooling layer
            emb = feature_extractor(img).flatten(start_dim=1)
            embeddings.append(emb.cpu()) # Move embeddings to CPU to save GPU memory
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    # embeddings = normalize(embeddings.cpu()) # Normalization might be needed depending on downstream task
    return embeddings, filenames

def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    # embeddings = normalize(embeddings.cpu())
    return embeddings, filenames



def generate_embeddings_from_model(model_path, config_path, output_dir):
    """
    Generate embeddings using a trained SimCLR model
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model checkpoint
    config_path : str
        Path to the config file used for training
    output_dir : str
        Directory to save the embeddings
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading data from: {config['data']['data_dir']}")
    data_dir = config['data']['data_dir']

    # Initialize data module and setup predict stage
    data_module = SimCLRDataModule(config)
    data_module.setup(stage="predict")  # Explicitly set the stage to "predict"
    print(f"Size of predict_dataset: {len(data_module.dataset_predict)}") 

    # Get the prediction dataloader
    dataloader = data_module.predict_dataloader()
    print(f"Size of dataloader: {len(dataloader)}")

    # Prepare model config
    model_config = {
        'scheduler': config['model']['scheduler'],
        'optimizer': config['model']['optimizer'],
        'criterion': config['model']['criterion'],
        'backbone': config['model']['backbone'],
        'weights': config['model']['weights']
    }
    
    # Load the trained model
    print(f"Loading model from checkpoint: {model_path}")
    model = SimCLRModel.load_from_checkpoint(
        model_path,
        config=model_config
    )
    model.eval()
    
    # print("Model's summary:")
    # print(model)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings, filenames = generate_embeddings_from_layer(model, dataloader)
    print("Embeddings generated.")
    
    # Save embeddings to output dir
    # pd.DataFrame(embeddings, index=filenames).to_csv(os.path.join(logger.log_dir, "embeddings.csv"), quoting=csv.QUOTE_ALL, encoding='utf-8')
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.columns = [f'x{i+1}' for i in range(len(df_embeddings.columns))]
    df_filenames = pd.DataFrame(filenames, columns=['filename'])
    df = pd.concat([df_filenames, df_embeddings], axis=1)

    # Extract the name of the config file without the directory path or extension
    config_filename = os.path.splitext(os.path.basename(config_path))[0]

    # Create a timestamped version string that includes the config filename
    date_str = dt.datetime.isoformat(dt.datetime.utcnow(), timespec='minutes')
    version = f"{config_filename}_{''.join(re.split(r'[-:]', date_str))}"

    output_filename = f"embeddings_{version}.csv"
    df.to_csv(
        os.path.join(output_dir, output_filename),
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8',
        float_format='%.15f')

    # Create sample plots
    print("Creating sample plots...")
    plot_knn_examples(embeddings, filenames, data_dir)

    # Save plots to output 
    plt.savefig(os.path.join(output_dir, "knn_examples.png"))

    # Perform Permanova analysis
    # print("Performing Permanova analysis...")
    # permanova_results = perform_permanova(embeddings, filenames)
    # with open(os.path.join(output_dir, "permanova_results.json"), 'w') as f:
    #     json.dump(permanova_results, f)



def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    generate_embeddings_from_model(
        args.model_path,
        args.config,
        args.output_dir
    )
    
    print("\nEmbeddings generation complete!")
  

def get_args():
    parser = ArgumentParser(description='Generate embeddings from a trained SimCLR model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the config file used for training')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the embeddings')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)