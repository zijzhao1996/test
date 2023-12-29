# Select the top 500 tickers based on their frequency
top_500_tickers = df['ticker'].value_counts().head(500).index.tolist()

# Filter for these tickers
df_filtered = df[df['ticker'].isin(top_500_tickers)]

# Filter for bar_times that end with '000' and are less than or equal to 153000
df_filtered = df_filtered[df_filtered['bar_time'].astype(str).str.endswith('000') & (df_filtered['bar_time'] <= 153000)]


import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
from functools import partial

def process_ticker_data(df, ticker, scale, seq_len):
    ticker_df = df[df['ticker'] == ticker]
    nb_obs = len(ticker_df)
    features = ticker_df[[col for col in ticker_df.columns if col.startswith('hist_return')]] * scale
    target = ticker_df['target'] * scale

    ticker_features = []
    ticker_targets = []

    for i in range(nb_obs - seq_len + 1):
        X_seq = torch.FloatTensor(features.iloc[i:i + seq_len].values)
        y_seq = torch.FloatTensor([target.iloc[i + seq_len - 1]])
        ticker_features.append(X_seq)
        ticker_targets.append(y_seq)

    return ticker_features, ticker_targets

def load_and_preprocess(years, scale=10000, seq_len=20):
    all_features = []
    all_targets = []

    for year in years:
        file_path = f'{year}_data.parquet'
        df = pd.read_parquet(file_path)
        tickers = df['ticker'].unique()

        with mp.Pool(mp.cpu_count()) as pool:
            func = partial(process_ticker_data, df, scale=scale, seq_len=seq_len)
            results = pool.map(func, tickers)

        for ticker_features, ticker_targets in results:
            all_features.extend(ticker_features)
            all_targets.extend(ticker_targets)

    features_tensor = torch.stack(all_features)
    targets_tensor = torch.stack(all_targets).squeeze(1)

    return features_tensor, targets_tensor

def create_dataloader(features, targets, batch_size=32, shuffle=True):
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Usage example
# TRAIN_YEARS = ['2008', '2009']
# VALID_YEARS = ['2010']
# train_features, train_targets = load_and_preprocess(TRAIN_YEARS)
# val_features, val_targets = load_and_preprocess(VALID_YEARS)
# train_loader = create_dataloader(train_features, train_targets)
# val_loader = create_dataloader(val_features, val_targets)





import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
from functools import partial
import os
import pickle

def process_ticker_data(df, ticker, scale, seq_len, temp_dir):
    ticker_df = df[df['ticker'] == ticker]
    nb_obs = len(ticker_df)
    features = ticker_df[[col for col in ticker_df.columns if col.startswith('hist_return')]] * scale
    target = ticker_df['target'] * scale

    ticker_features = []
    ticker_targets = []

    for i in range(nb_obs - seq_len + 1):
        X_seq = torch.FloatTensor(features.iloc[i:i + seq_len].values)
        y_seq = torch.FloatTensor([target.iloc[i + seq_len - 1]])
        ticker_features.append(X_seq)
        ticker_targets.append(y_seq)

    # Save processed data to disk
    file_path = os.path.join(temp_dir, f"{ticker}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump((ticker_features, ticker_targets), f)

    print(f"Processed ticker {ticker}: {len(ticker_features)} sequences.")

def load_and_preprocess(years, scale=10000, seq_len=20, temp_dir="temp_data"):
    all_features = []
    all_targets = []

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for year in years:
        file_path = f'{year}_data.parquet'
        df = pd.read_parquet(file_path)
        tickers = df['ticker'].unique()

        with mp.Pool(mp.cpu_count()) as pool:
            func = partial(process_ticker_data, df, scale=scale, seq_len=seq_len, temp_dir=temp_dir)
            pool.map(func, tickers)

        # Read processed data from disk
        for ticker in tickers:
            file_path = os.path.join(temp_dir, f"{ticker}.pkl")
            with open(file_path, 'rb') as f:
                ticker_features, ticker_targets = pickle.load(f)
                all_features.extend(ticker_features)
                all_targets.extend(ticker_targets)
            os.remove(file_path)  # Optionally remove the file after loading

    features_tensor = torch.stack(all_features)
    targets_tensor = torch.stack(all_targets).squeeze(1)

    return features_tensor, targets_tensor

# Rest of the code remains the same











import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
from functools import partial
import os
import pickle

def process_ticker_data(df, ticker, scale, seq_len, year_temp_dir):
    ticker_df = df[df['ticker'] == ticker]
    nb_obs = len(ticker_df)
    features = ticker_df[[col for col in ticker_df.columns if col.startswith('hist_return')]] * scale
    target = ticker_df['target'] * scale

    ticker_features = []
    ticker_targets = []

    for i in range(nb_obs - seq_len + 1):
        X_seq = torch.FloatTensor(features.iloc[i:i + seq_len].values)
        y_seq = torch.FloatTensor([target.iloc[i + seq_len - 1]])
        ticker_features.append(X_seq)
        ticker_targets.append(y_seq)

    # Save processed data to disk
    file_path = os.path.join(year_temp_dir, f"{ticker}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump((ticker_features, ticker_targets), f)

    print(f"Processed ticker {ticker}: {len(ticker_features)} sequences.")

def load_and_preprocess(years, scale=10000, seq_len=20, temp_dir="temp_data"):
    all_features = []
    all_targets = []

    for year in years:
        year_temp_dir = os.path.join(temp_dir, year)
        if not os.path.exists(year_temp_dir):
            os.makedirs(year_temp_dir)

        file_path = f'{year}_data.parquet'
        df = pd.read_parquet(file_path)
        tickers = df['ticker'].unique()

        with mp.Pool(mp.cpu_count()) as pool:
            func = partial(process_ticker_data, df, scale=scale, seq_len=seq_len, year_temp_dir=year_temp_dir)
            pool.map(func, tickers)

        # Read processed data from disk
        for ticker in tickers:
            file_path = os.path.join(year_temp_dir, f"{ticker}.pkl")
            with open(file_path, 'rb') as f:
                ticker_features, ticker_targets = pickle.load(f)
                all_features.extend(ticker_features)
                all_targets.extend(ticker_targets)
            os.remove(file_path)  # Optionally remove the file after loading

    features_tensor = torch.stack(all_features)
    targets_tensor = torch.stack(all_targets).squeeze(1)

    return features_tensor, targets_tensor

def create_dataloader(features, targets, batch_size=32, shuffle=True):
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Usage example
# TRAIN_YEARS = ['2008', '2009']
# VALID_YEARS = ['2010']
# train_features, train_targets = load_and_preprocess(TRAIN_YEARS)
# val_features, val_targets = load_and_preprocess(VALID_YEARS)
# train_loader = create_dataloader(train_features, train_targets)
# val_loader = create_dataloader(val_features, val_targets)


