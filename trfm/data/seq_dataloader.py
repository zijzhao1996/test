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

    # Filter out rows where target/features are None
    selected_cols = ['target'] + [col for col in df.columns if col.startswith('hist_return')]
    df = df[selected_cols].dropna()

    # Process data in parallel by ticker
    tickers = df['ticker'].unique().tolist()
    with multiprocessing.Pool() as pool:
        pool.starmap(dump_seq_data_per_ticker, [(df, ticker, scale, seq_len, year_temp_dir) for ticker in tickers])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    dump_seq_data('2008', scale=1e4, seq_len=10)