import os
import pandas as pd
import argparse
from trfm.trainer import Trainer
from test.trfm.data.nonseq_dataloader import create_dataloader


def main(config_path):
    # Load configuration
    trainer = Trainer(config_path)

    # Load data
    train_df = pd.read_parquet('/dat/chbr_group/chbr_scratch/non_sequential_data/2008_data.parquet')
    valid_df = pd.read_parquet('/dat/chbr_group/chbr_scratch/non_sequential_data/2009_data.parquet')

    # Create DataLoaders
    train_dataloader = create_dataloader(dataframe=train_df,
                                        year='2008',
                                        batch_size=trainer.config['training_params']['batch_size'],
                                        shuffle=True,
                                        scale=1e4,
                                        downsample=True)
    valid_dataloader = create_dataloader(dataframe=valid_df,
                                        year='2009',
                                        batch_size=trainer.config['training_params']['batch_size'],
                                        shuffle=False,
                                        scale=1e4,
                                        downsample=True)

    # Train the model
    trainer.train(train_dataloader, valid_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config)