import warnings
warnings.filterwarnings("ignore")

import os
import sys
# Add the parent directory of 'train' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import torch
import yaml
import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl
# from simclr_birdcolour import SimCLRModel, SimCLRDataModule
# from simclr_birdcolour_tune_eval import perform_permanova, generate_embeddings, plot_knn_examples
from simclr_birdcolour_kornia02 import SimCLRModel, SimCLRDataModule, generate_embeddings, plot_knn_examples
import datetime as dt
import re
import csv
import matplotlib.pyplot as plt
import json

def generate_embeddings_from_model(model_path, config_path, output_dir, data_dir):
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
    data_dir : str
        Path to the data directory
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading data from: {data_dir}")

    # Override the data_dir in config with the provided argument
    config['data']['data_dir'] = data_dir
    
    # Fix PyTorch DataLoader shared memory error by disabling multiprocessing
    if 'dataloader' not in config:
        config['dataloader'] = {}
    config['dataloader']['num_workers'] = 0
    print("Set num_workers=0 to avoid shared memory issues during embedding generation")

    # Initialize data module and setup predict stage
    data_module = SimCLRDataModule(config)
    data_module.setup(stage="predict")  # Explicitly set the stage to "predict"
    
    # Get the prediction dataloader
    dataloader = data_module.predict_dataloader()
    
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
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings, filenames = generate_embeddings(model, dataloader)
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
    date_str = dt.datetime.isoformat(dt.datetime.utcnow() + dt.timedelta(hours=1), timespec='minutes')
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
        args.output_dir,
        args.data_dir
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
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the data directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)