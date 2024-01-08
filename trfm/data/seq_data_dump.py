import os
import pandas as pd
import torch
import pickle
import multiprocessing
import logging
from torch.utils.data import Dataset

def dump_seq_data_per_ticker(df, ticker, scale, seq_len, year_temp_dir):
    """
    Process and save the sequential data for a specific ticker.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    ticker (str): Ticker symbol to process.
    scale (float): Scaling factor for data normalization.
    seq_len (int): Length of the sequential data window.
    year_temp_dir (str): Directory to save processed data.
    """
    output_file = os.path.join(year_temp_dir, f"{ticker}.pkl")

    # check if the output file for this ticker already exists
    if os.path.exists(output_file):
        logging.info(f"File for ticker {ticker} already exists. Skipping processing.")
        return

    ticker_df = df[df['ticker'] == ticker]
    # Make sure they are sorted by date & bar_time
    ticker_df = ticker_df.sort_values(['date', 'bar_time']).reset_index(drop=True)
    features = ticker_df.filter(like='hist_ret').values * scale
    target = ticker_df['target'].values * scale

    ticker_features, ticker_targets = [], []
    for i in range(len(ticker_df) - seq_len + 1):
        X_seq = torch.FloatTensor(features[i:i + seq_len])
        y_seq = torch.FloatTensor([target[i + seq_len - 1]])
        ticker_features.append(X_seq)
        ticker_targets.append(y_seq)

    # Save processed data to disk
    with open(output_file, 'wb') as f:
        pickle.dump((ticker_features, ticker_targets), f)

    logging.info(f"Processed ticker {ticker}: {len(ticker_features)} sequences.")

def dump_seq_data(year, scale=1, seq_len=10, temp_dir="/dat/chbr_group/chbr_scratch/sequential_data_temp", downsample=False):
    """
    Process and save the sequential data for a given year.

    Args:
    year (str): Year for which to process data.
    scale (float): Scaling factor for data normalization.
    seq_len (int): Length of the sequential data window.
    temp_dir (str): Temporary directory for processed data.
    downsample (bool): Flag to downsample the data.
    """
    year_temp_dir = os.path.join(temp_dir, f"{year}/seq_len_{seq_len}")  # Include seq_len in path
    if not os.path.exists(year_temp_dir):
        os.makedirs(year_temp_dir)

    file_path = f'/dat/chbr_group/chbr_scratch/non_sequential_data/{year}_data.parquet'
    df = pd.read_parquet(file_path)

    if downsample:
        df = df[df['bar_time'].astype(str).str.endswith('0000')]

    # Process data in parallel by ticker
    tickers = df['ticker'].unique().tolist()

    # Filter out rows where target/features are None
    selected_cols = ['target'] + [col for col in df.columns if col.startswith('hist_ret')]
    df.dropna(subset=selected_cols, inplace=True)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(dump_seq_data_per_ticker, [(df, ticker, scale, seq_len, year_temp_dir) for ticker in tickers])


def load_temp_data(year, base_temp_dir='/dat/chbr_group/chbr_scratch/sequential_data_temp',
                save_dir='/dat/chbr_group/chbr_scratch/sequential_data',
                seq_len=10):
    """
    Load temporary sequential data per ticker, concatenate it, and save as a single PyTorch tensor.

    Args:
    year (str): The year for which to process the data.
    base_temp_dir (str): Directory where the temporary per-ticker data is stored.
    save_dir (str): Directory where the concatenated data will be saved.

    Returns:
    str: The file path to the saved concatenated data.
    """
    year_temp_dir = os.path.join(base_temp_dir, f"{year}/seq_len_{seq_len}")  # Include seq_len in path
    all_features, all_targets = [], []
    tickers = [f for f in os.listdir(year_temp_dir) if f.endswith('.pkl')]

    # Load and concatenate data from each ticker
    for ticker in tickers:
        file_path = os.path.join(year_temp_dir, ticker)
        with open(file_path, 'rb') as f:
            ticker_features, ticker_targets = pickle.load(f)
            all_features.extend(ticker_features)
            all_targets.extend(ticker_targets)
        logging.info(f"Loaded data for ticker {ticker}.")

    # Convert lists of features and targets to tensors
    features_tensor = torch.stack(all_features)
    targets_tensor = torch.stack(all_targets).squeeze(1)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the concatenated features and targets
    dataset_file_path = os.path.join(save_dir, f'{year}_{seq_len}_final_dataset.pt')
    torch.save({'features': features_tensor, 'labels': targets_tensor}, dataset_file_path)

    logging.info(f"Concatenated data for year {year} saved at {dataset_file_path}")
    return dataset_file_path

class SeqDataset(Dataset):
    """
    PyTorch Dataset class for loading sequential data from pre-processed tensors.
    """
    def __init__(self, dataset_file):
        """
        Initialize the dataset.

        Args:
        dataset_file (str): Path to the file containing the pre-processed tensors.
        """
        data = torch.load(dataset_file)
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Fetches the features and label for the specified index."""
        return self.features[idx], self.labels[idx]