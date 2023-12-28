import pandas as pd
import os
import concurrent.futures

def get_previous_n_days_inclusive(target_date, n):
    # Implementation assumed to be provided.
    pass

def process_date(file_path, path):
    # Extract date from file name
    target_date = os.path.basename(file_path).split('.')[0]

    # Get the dates for the target date and the previous 10 days
    dates_to_load = get_previous_n_days_inclusive(target_date, 10)

    # Check if we have enough data (i.e., at least 10 days including the target date)
    if len(dates_to_load) < 11:
        print(f"Not enough data for date: {target_date}. Skipping.")
        return None

    # Filter and read the relevant files
    files_to_read = [os.path.join(path, f"{date}.parquet") for date in dates_to_load if os.path.exists(os.path.join(path, f"{date}.parquet"))]
    df = pd.concat([pd.read_parquet(file) for file in files_to_read])

    # Ensure the dataframe is sorted by ticker and bar_time
    df.sort_values(by=['ticker', 'bar_time'], inplace=True)

    # Define time intervals in terms of 5-minute bars
    intervals = {'5mins': 1, '15mins': 3, '30mins': 6, '1hour': 12, '3hours': 36,
                 '5hours': 60, '1day': 12*13, '3days': 12*13*3, '5days': 12*13*5, '10days': 12*13*10}

    # Compute historical returns for each interval
    for interval_name, interval_bars in intervals.items():
        df[f'hist_ret_{interval_name}'] = df.groupby('ticker')['ret'].apply(
            lambda x: x.iloc[::-1].rolling(window=interval_bars, min_periods=1).apply(
                lambda y: (y + 1).prod() - 1, raw=True).iloc[::-1])

    print(f"Processed data for date: {target_date}")
    return df

def obtain_features(path):
    # Get a list of all parquet files in the directory
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]

    # Process each file in parallel using multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(lambda file: process_date(file, path), files)

    # Filter out None results and concatenate all dataframes
    combined_df = pd.concat([df for df in results if df is not None], ignore_index=True)

    return combined_df

# Example usage
path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'
features_df = obtain_features(path)







import pandas as pd
import os
import multiprocessing

def get_previous_n_days_inclusive(target_date, n):
    # Implementation assumed to be provided.
    pass

def process_date(file_path, path):
    # Extract date from file name
    target_date = os.path.basename(file_path).split('.')[0]

    # Get the dates for the target date and the previous 10 days
    dates_to_load = get_previous_n_days_inclusive(target_date, 10)

    # Check if we have enough data
    if len(dates_to_load) < 11:
        print(f"Not enough data for date: {target_date}. Skipping.")
        return None

    # Filter and read the relevant files
    files_to_read = [os.path.join(path, f"{date}.parquet") for date in dates_to_load if os.path.exists(os.path.join(path, f"{date}.parquet"))]
    df = pd.concat([pd.read_parquet(file) for file in files_to_read])

    # Ensure the dataframe is sorted by ticker and bar_time
    df.sort_values(by=['ticker', 'bar_time'], inplace=True)

    # Define time intervals in terms of 5-minute bars
    intervals = {'5mins': 1, '15mins': 3, '30mins': 6, '1hour': 12, '3hours': 36, '5hours': 60, '1day': 12*13, '3days': 12*13*3, '5days': 12*13*5, '10days': 12*13*10}

    # Compute historical returns for each interval
    for interval_name, interval_bars in intervals.items():
        df[f'hist_ret_{interval_name}'] = df.groupby('ticker')['ret'].apply(
            lambda x: x.iloc[::-1].rolling(window=interval_bars, min_periods=1).apply(
                lambda y: (y + 1).prod() - 1, raw=True).iloc[::-1])

    print(f"Processed data for date: {target_date}")
    return df

def obtain_features(path):
    # Get a list of all parquet files in the directory
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]

    # Create a pool of workers and process each file
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_date, [(file, path) for file in files])

    # Filter out None results and concatenate all dataframes
    combined_df = pd.concat([df for df in results if df is not None], ignore_index=True)

    return combined_df

# Example usage
path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'
features_df = obtain_features(path)

