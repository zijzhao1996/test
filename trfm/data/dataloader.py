import os
import torch
import logging
import random
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
from trfm.data.seq_data_dump import dump_seq_data, load_temp_data, SeqDataset
from trfm.data.nonseq_data_dump import NoseqDataset


def save_dataset(dataset, file_path):
    """Saves the dataset to a file."""
    torch.save({'features': dataset.features, 'labels': dataset.labels}, file_path)

def load_dataset(file_path):
    """Loads the dataset from a file."""
    data = torch.load(file_path)
    features_tensor = data['features'].clone().detach()
    labels_tensor = data['labels'].clone().detach()
    return TensorDataset(features_tensor, labels_tensor)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(years, batch_size=32, shuffle=True, scale=1, seq_len=10, downsample=False, is_seq=False, dataframe=None):
    """
    Creates a DataLoader from the given DataFrame for multiple years.

    Args:
    years (list of str): Years associated with the data.
    batch_size (int): Batch size for the DataLoader.
    shuffle (bool): Whether to shuffle the data.
    scale (float): Scaling factor for the data.
    downsample (bool): If True, downsample the data.
    is_seq (bool): Whether the data is sequential.
    dataframe (pd.DataFrame): Source data.

    Returns:
    DataLoader: The DataLoader object for the dataset.
    """
    # Initialize lists to store features and labels
    all_features, all_labels = [], []

    # Form the dataset file name based on the years
    years_str = '_'.join(years)
    dataset_file_name = f'{years_str}_seq{seq_len}_final_dataset.pt' if is_seq else f'{years_str}_final_dataset.pt'
    dataset_file_path = os.path.join('/dat/chbr_group/chbr_scratch/', 'sequential_data' if is_seq else 'non_sequential_data', dataset_file_name)

    for year in years:
        # Determine file paths based on whether data is sequential or not
        if not is_seq:
            assert dataframe is not None, 'Dataframe must be provided for non-sequential data.'
            assert seq_len is None, 'Sequence length must be None for non-sequential data.'

        # Load or create dataset for each year
        if os.path.exists(dataset_file_path):
            dataset = load_dataset(dataset_file_path)
            logging.info(f'File found. Loaded dataset from {dataset_file_path}')
        else:
            # Handle data processing based on sequential or non-sequential data
            if is_seq:
                data_dir = f'/dat/chbr_group/chbr_scratch/sequential_data_temp/{year}/seq{seq_len}'
                if not os.path.exists(data_dir):
                    dump_seq_data(year, scale=scale, seq_len=seq_len, downsample=downsample)
                data_file = load_temp_data(year, seq_len=seq_len)
                dataset = SeqDataset(data_file)
                save_dataset(dataset, dataset_file_path)
            else:
                logging.info(f'Creating dataset for year {year}.')
                dataset = NoseqDataset(dataframe, scale=scale, downsample=downsample)
                save_dataset(dataset, dataset_file_path)

        # Append features and labels to the lists
        all_features.append(dataset.features)
        all_labels.append(dataset.labels)

    # Concatenate all features and labels
    concatenated_features = torch.cat(all_features, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)

    # Create a final dataset from concatenated features and labels
    final_dataset = TensorDataset(concatenated_features, concatenated_labels)

    # Use maximum number of workers
    num_workers = multiprocessing.cpu_count()

    # Setup for reproducible shuffling using torch.Generator
    g = torch.Generator()
    g.manual_seed(0)

    # Create the DataLoader
    dataloader = DataLoader(
        final_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g if shuffle else None
    )
    return dataloader
