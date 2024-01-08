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
    features_tensor = torch.tensor(data['features'], dtype=torch.float32)
    labels_tensor = torch.tensor(data['labels'], dtype=torch.float32)
    return TensorDataset(features_tensor, labels_tensor)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(year, batch_size=32, shuffle=True, scale=1, seq_len=10, downsample=False, is_seq=False, dataframe=None):
    """
    Creates a DataLoader from the given DataFrame.

    Args:
    year (str): Year associated with the data.
    batch_size (int): Batch size for the DataLoader.
    shuffle (bool): Whether to shuffle the data.
    scale (float): Scaling factor for the data.
    downsample (bool): If True, downsample the data.
    dataframe (pd.DataFrame): Source data.

    Returns:
    DataLoader: The DataLoader object for the dataset.
    """
    if is_seq:
        dataset_file_path = f'/dat/chbr_group/chbr_scratch/sequential_data/{year}_seq{seq_len}_final_dataset.pt'
    else:
        dataset_file_path = f'/dat/chbr_group/chbr_scratch/non_sequential_data/{year}_final_dataset.pt'
        assert dataframe is not None, 'Dataframe must be provided for non-sequential data.'
        assert seq_len is None, 'Sequence length must be None for non-sequential data.'

    # Check if dataset file exists
    if os.path.exists(dataset_file_path):
        dataset = load_dataset(dataset_file_path)
        logging.info(f'File found. Loaded dataset from {dataset_file_path}')
    else:
        if is_seq:
            data_dir = f'/dat/chbr_group/chbr_scratch/sequential_data_temp/{year}/seq{seq_len}'
            if not os.path.exists(data_dir):
                dump_seq_data(year, scale=scale, seq_len=seq_len, downsample=downsample)
            data_file = load_temp_data(year, seq_len=seq_len)
            dataset = SeqDataset(data_file)
            save_dataset(dataset, dataset_file_path)
        else:
            # Create and save the dataset
            logging.info(f'Creating dataset for year {year}.')
            dataset = NoseqDataset(dataframe, scale=scale, downsample=downsample)
            save_dataset(dataset, dataset_file_path)

    # Use maximum number of workers
    num_workers = multiprocessing.cpu_count()

    # Setup for reproducible shuffling using torch.Generator
    g = torch.Generator()
    g.manual_seed(0)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=g if shuffle else None
    )
    return dataloader
