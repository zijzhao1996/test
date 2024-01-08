import pandas as pd
import argparse
from trfm.trainer import Trainer
from trfm.data.dataloader import create_dataloader

def main(config_path):
    # Load configuration
    trainer = Trainer(config_path)

    # Extract training and validation years from the configuration
    train_year = trainer.config['data_params']['train_year']
    valid_year = trainer.config['data_params']['valid_year']
    is_seq = trainer.config['data_params']['is_seq']

    # Load dataframe based on the type of dataset
    if is_seq:
        # Handle non-sequential data loading if necessary
        train_df = None
        valid_df = None
    else:
        train_df = pd.read_parquet(f'/dat/chbr_group/chbr_scratch/non_sequential_data/{train_year}_data.parquet')
        valid_df = pd.read_parquet(f'/dat/chbr_group/chbr_scratch/non_sequential_data/{valid_year}_data.parquet')

    # Create DataLoaders
    seq_len = trainer.config['data_params']['seq_len'] if is_seq else None
    train_dataloader = create_dataloader(year=train_year,
                                        batch_size=trainer.config['training_params']['batch_size'],
                                        shuffle=True,
                                        scale=1e4,
                                        seq_len=seq_len,
                                        downsample=True,
                                        is_seq=is_seq,
                                        dataframe=train_df)
    valid_dataloader = create_dataloader(year=valid_year,
                                        batch_size=trainer.config['training_params']['batch_size'],
                                        shuffle=False,
                                        scale=1e4,
                                        seq_len=seq_len,
                                        downsample=True,
                                        is_seq=is_seq,
                                        dataframe=valid_df)

    # Train the model
    trainer.train(train_dataloader, valid_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config)
