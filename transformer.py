import pandas as pd
import os
import concurrent.futures
from datetime import datetime

def get_previous_n_days_inclusive(target_date, n):
    # This function returns a list of dates (as strings) for the target date and the previous n days.
    # Implementation is assumed to be provided.
    pass

def process_date(target_date, path):
    # Get the dates for the target date and the previous 10 days
    dates_to_load = get_previous_n_days_inclusive(target_date, 10)

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

def obtain_features(path, start_date, end_date):
    # Generate a list of dates to process
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strings = [date.strftime('%Y%m%d') for date in date_range]

    # Process each date in parallel using multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(lambda date: process_date(date, path), date_strings)

    # Concatenate all dataframes
    combined_df = pd.concat(results, ignore_index=True)

    return combined_df

# Example usage
path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'
start_date = '20180601'
end_date = '20180630'
features_df = obtain_features(path, start_date, end_date)
