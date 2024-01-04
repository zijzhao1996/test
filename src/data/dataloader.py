import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class NoseqDataset(Dataset):
    def __init__(self, dataframe, scale=1, downsample=False):
        """
        Initialize the dataset.

        Args:
        dataframe (pd.DataFrame): Source data.
        scale (float): Scaling factor for the data.
        downsample (bool): If True, downsample the data.
        """
        self.dataframe = dataframe
        if downsample:
            self.dataframe = self.dataframe[self.dataframe['bar_time'].astype(str).str.endswith('000')]

        selected_cols = ['target'] + [col for col in self.dataframe.columns if col.startswith('hist_ret')]
        self.dataframe = self.dataframe[selected_cols].dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        self.features = self.dataframe.filter(like='hist_ret').values * scale
        self.labels = self.dataframe['target'].values * scale

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Fetches the features and label for a given index."""
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)  # Shape [1]
        return features, label


def save_dataset(dataset, file_path):
    """Saves the dataset to a file."""
    torch.save({'features': dataset.features, 'labels': dataset.labels}, file_path)

def load_dataset(file_path):
    """Loads the dataset from a file."""
    data = torch.load(file_path)
    return TensorDataset(torch.tensor(data['features'], dtype=torch.float32),
                         torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(-1))

def create_dataloader(dataframe, year, batch_size=32, shuffle=True, scale=1, downsample=False):
    """
    Creates a DataLoader from the given DataFrame.

    Args:
    dataframe (pd.DataFrame): Source data.
    year (str): Year associated with the data.
    batch_size (int): Batch size for the DataLoader.
    shuffle (bool): Whether to shuffle the data.
    scale (float): Scaling factor for the data.
    downsample (bool): If True, downsample the data.

    Returns:
    DataLoader: The DataLoader object for the dataset.
    """
    dataset_file_path = f'/dat/chbr_group/chbr_scratch/non_seqential_dataset/{year}_data.pt'

    # Check if dataset file exists
    if os.path.exists(dataset_file_path):
        dataset = load_dataset(dataset_file_path)
        logging.info(f'File found. Loaded dataset from {dataset_file_path}')
    else:
        # Create and save the dataset
        logging.info(f'Creating dataset for year {year}.')
        dataset = NoseqDataset(dataframe, scale=scale, downsample=downsample)
        save_dataset(dataset, dataset_file_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
