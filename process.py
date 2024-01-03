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






import os
import pickle
import torch

def load_temp_data(year_temp_dir):
    all_features = []
    all_targets = []
    tickers = [f.split('.')[0] for f in os.listdir(year_temp_dir) if f.endswith('.pkl')]

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

# Now you can call load_temp_data independently
# Example:
# year_temp_dir = "temp_data/2008"
# features_tensor, targets_tensor = load_temp_data(year_temp_dir)






import os
import pickle
import torch

def save_combined_dataset(features, targets, file_path):
    torch.save((features, targets), file_path)

def load_combined_dataset(file_path):
    return torch.load(file_path)

def combined_dataset_file_path(year, save_dir="combined_data"):
    return os.path.join(save_dir, f"combined_dataset_{year}.pt")

def load_temp_data(years, temp_dir="temp_data", save_dir="combined_data"):
    all_features = []
    all_targets = []

    for year in years:
        year_combined_file = combined_dataset_file_path(year, save_dir)
        
        # Check if the combined dataset for the year already exists
        if os.path.exists(year_combined_file):
            print(f"Loading combined dataset for year {year} from {year_combined_file}")
            features_tensor, targets_tensor = load_combined_dataset(year_combined_file)
        else:
            print(f"Processing temporary data for year {year}...")
            year_temp_dir = os.path.join(temp_dir, year)
            file_names = [f for f in os.listdir(year_temp_dir) if f.endswith('.pkl')]

            year_features = []
            year_targets = []

            for file_name in file_names:
                file_path = os.path.join(year_temp_dir, file_name)
                with open(file_path, 'rb') as f:
                    ticker_features, ticker_targets = pickle.load(f)
                year_features.extend(ticker_features)
                year_targets.extend(ticker_targets)
                os.remove(file_path)  # Optionally remove the file after loading

            features_tensor = torch.stack(year_features)
            targets_tensor = torch.stack(year_targets).squeeze(1)
            
            # Save the combined dataset for the year
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_combined_dataset(features_tensor, targets_tensor, year_combined_file)
            print(f"Saved combined dataset for year {year} to {year_combined_file}")

        all_features.append(features_tensor)
        all_targets.append(targets_tensor)

    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

# Example usage:
# TRAIN_YEARS = ['2008', '2009']
# train_features, train_targets = load_temp_data(TRAIN_YEARS)





import os
import pickle
import torch

def save_combined_dataset(features, targets, file_path):
    torch.save((features, targets), file_path)

def load_combined_dataset(file_path):
    return torch.load(file_path)

def combined_dataset_file_path(years, save_dir="combined_data"):
    years_str = '_'.join(years)
    return os.path.join(save_dir, f"combined_dataset_{years_str}.pt")

def load_temp_data(years, temp_dir="temp_data", save_dir="combined_data"):
    all_features = []
    all_targets = []

    combined_file_path = combined_dataset_file_path(years, save_dir)

    # Check if the combined dataset for the years already exists
    if os.path.exists(combined_file_path):
        print(f"Loading combined dataset from {combined_file_path}")
        features_tensor, targets_tensor = load_combined_dataset(combined_file_path)
        return features_tensor, targets_tensor

    for year in years:
        year_temp_dir = os.path.join(temp_dir, year)
        file_names = [f for f in os.listdir(year_temp_dir) if f.endswith('.pkl')]

        for file_name in file_names:
            file_path = os.path.join(year_temp_dir, file_name)
            with open(file_path, 'rb') as f:
                ticker_features, ticker_targets = pickle.load(f)
            all_features.extend(ticker_features)
            all_targets.extend(ticker_targets)
            os.remove(file_path)  # Optionally remove the file after loading

    # Combine and save the dataset
    combined_features_tensor = torch.stack(all_features)
    combined_targets_tensor = torch.stack(all_targets).squeeze(1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_combined_dataset(combined_features_tensor, combined_targets_tensor, combined_file_path)
    print(f"Saved combined dataset to {combined_file_path}")

    return combined_features_tensor, combined_targets_tensor

# Example usage:
# TRAIN_YEARS = ['2008', '2009']
# train_features, train_targets = load_temp_data(TRAIN_YEARS)

def process_ticker_data(file_path, ticker, scale, seq_len, year_temp_dir):
    df = pd.read_parquet(file_path)
    ticker_df = df[df['ticker'] == ticker]
    # ... rest of your processing logic ...

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
            func = partial(process_ticker_data, file_path, scale=scale, seq_len=seq_len, year_temp_dir=year_temp_dir)
            pool.map(func, tickers)








import torch
from torch.utils.data import Dataset

class FinancialDataset(Dataset):
    def __init__(self, features_tensor, targets_tensor):
        """
        Initializes the dataset with features and targets tensors.
        Args:
            features_tensor (torch.Tensor): The tensor containing features.
            targets_tensor (torch.Tensor): The tensor containing targets.
        """
        self.features = features_tensor
        self.labels = targets_tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a single data sample at given index.
        Args:
            idx (int): The index of the sample to return.
        Returns:
            tuple: (feature, label) tensors.
        """
        features = self.features[idx]
        label = self.labels[idx]
        return features, label

# Usage example
# Load data using load_temp_data function
year_temp_dir = "temp_data/2008"
features_tensor, targets_tensor = load_temp_data(year_temp_dir)

# Create the dataset
financial_dataset = FinancialDataset(features_tensor, targets_tensor)

# Now you can use financial_dataset with a DataLoader, for example
# train_loader = DataLoader(financial_dataset, batch_size=32, shuffle=True)
