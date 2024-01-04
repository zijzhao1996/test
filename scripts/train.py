import os
import pandas as pd
from src.trainer import Trainer

def main():
    config_path = os.path.join(os.getcwd(), 'config', 'linear_model.yaml')
    trainer = Trainer(config_path)

    # Load data
    train_dataloader = pd.read_parquet('/dat/chbr_group/chbr_scratch/non_sequential_data/2008_data.parquet')
    valid_dataloader = pd.read_parquet('/dat/chbr_group/chbr_scratch/non_sequential_data/2009_data.parquet')

    trainer.train(train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()