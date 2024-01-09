import pandas as pd
import torch
from torch.utils.data import Dataset

class NoseqDataset(Dataset):
    def __init__(self, year, scale=1, downsample=False, base_dir='/dat/chbr_group/chbr_scratch/non_sequential_data'):
        """
        Initialize the dataset.

        Args:
        year (str): Year of the data to load.
        scale (float): Scaling factor for the data.
        downsample (bool): If True, downsample the data.
        base_dir (str): Base directory where the data files are stored.
        """
        file_path = f'{base_dir}/{year}_data.parquet'
        self.dataframe = pd.read_parquet(file_path)

        if downsample:
            self.dataframe = self.dataframe[self.dataframe['bar_time'].astype(str).str.endswith('0000')]

        selected_cols = ['target'] + [col for col in self.dataframe.columns if col.startswith('hist_ret')]
        self.dataframe = self.dataframe[selected_cols].dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        self.features = torch.tensor(self.dataframe.filter(like='hist_ret').values * scale, dtype=torch.float32)
        self.labels = torch.tensor(self.dataframe['target'].values * scale, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Fetches the features and label for a given index."""
        return self.features[idx], self.labels[idx]