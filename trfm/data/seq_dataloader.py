import os
import pandas as pd
import torch
import pickle
import multiprocessing
import logging

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
    year_temp_dir = os.path.join(temp_dir, year)
    if not os.path.exists(year_temp_dir):
        os.makedirs(year_temp_dir)

    file_path = f'/dat/chbr_group/chbr_scratch/non_sequential_data/{year}_data.parquet'
    df = pd.read_parquet(file_path)

    if downsample:
        df = df[df['bar_time'].astype(str).str.endswith('0000')]

    # Process data in parallel by ticker
    tickers = df['ticker'].unique().tolist()

    # Filter out rows where target/features are None
    selected_cols = ['target'] + [col for col in df.columns if col.startswith('hist_return')]
    df.dropna(subset=selected_cols, inplace=True)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(dump_seq_data_per_ticker, [(df, ticker, scale, seq_len, year_temp_dir) for ticker in tickers])


def load_temp_data(year, 
                   base_temp_dir='/dat/chbr_group/chbr_scratch/sequential_data_temp', 
                   save_dir='/dat/chbr_group/chbr_scratch/sequential_data'):
    """
    Load temporary data for a given year, concatenate it, and save as PyTorch tensors.

    Args:
    year (str): Year for which to load and process data.
    base_temp_dir (str): Directory containing temporary per-ticker data.
    save_dir (str): Directory to save concatenated data.

    Returns:
    Tuple[str, str]: Paths to the saved feature and target tensors.
    """
    year_temp_dir = os.path.join(base_temp_dir, year)
    all_features, all_targets = [], []
    tickers = [f for f in os.listdir(year_temp_dir) if f.endswith('.pkl')]

    for ticker in tickers:
        file_path = os.path.join(year_temp_dir, ticker)
        with open(file_path, 'rb') as f:
            ticker_features, ticker_targets = pickle.load(f)
            all_features.extend(ticker_features)
            all_targets.extend(ticker_targets)
        logging.info(f"Loaded ticker {ticker}.")

    features_tensor = torch.stack(all_features)
    targets_tensor = torch.stack(all_targets).squeeze(1)

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the processed tensors
    feature_file = os.path.join(save_dir, f'features_{year}.pt')
    target_file = os.path.join(save_dir, f'targets_{year}.pt')
    torch.save(features_tensor, feature_file)
    torch.save(targets_tensor, target_file)

    logging.info(f"Concatenated data saved for year {year}")
    return feature_file, target_file

class SeqDataset(Dataset):
    """
    PyTorch Dataset class for loading sequential data.
    """
    def __init__(self, feature_file, target_file):
        """
        Initialize the dataset with paths to the feature and target files.

        Args:
        feature_file (str): Path to the feature tensor file.
        target_file (str): Path to the target tensor file.
        """
        self.features = torch.load(feature_file)
        self.labels = torch.load(target_file)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Fetches the features and label for a given index."""
        return self.features[idx], self.labels[idx]




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    dump_seq_data('2008', scale=1e4, seq_len=10, downsample=True)
    dump_seq_data('2009', scale=1e4, seq_len=10, downsample=True) #TODO: save different se len and adapt name
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    feature_file, target_file = load_temp_data('2008')
    dataset = SeqDataset(feature_file, target_file)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)