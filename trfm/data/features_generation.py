import pandas as pd
import numpy as np
from datetime import datetime
import os
import multiprocessing
import logging
from utils.tradedays import get_previous_n_days

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_features_single_date(file_path, path):
    """
    Compute features for a single date from the given file.

    Args:
    file_path (str): The path to the parquet file to process.
    path (str): The directory path containing the trading data files.

    Returns:
    pd.DataFrame: A DataFrame with computed features for the given date.
    """
    try:
        # Extracting the target date from the file name
        target_date_str = os.path.basename(file_path).split('.')[0]
        target_date = datetime.strptime(target_date_str, '%Y%m%d').strftime('%Y-%m-%d')

        # Define the lookback period
        N = 3
        # Get the dates for the target date and the previous N days
        dates_to_load = get_previous_n_days(target_date_str, N, path)

        # Ensure we have enough data to proceed
        if len(dates_to_load) < N:
            logging.warning(f"Not enough data for date: {target_date}. Skipping.")
            return None

        # Read relevant files for the specified dates
        files_to_read = [os.path.join(path, f"{date}.parquet") for date in dates_to_load if os.path.exists(os.path.join(path, f"{date}.parquet"))]
        df = pd.concat([pd.read_parquet(file) for file in files_to_read], ignore_index=True)

        # Handling outliers in the 'ret' column
        if df['ret'].max() > 10:
            logging.warning(f"Outlier detected in {file_path}")
            df.loc[df['ret'] > 10, 'ret'] = np.nan

        # Preparing the DataFrame for feature calculation
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['ticker', 'date'], inplace=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # Define time intervals for historical return calculation
        intervals = {'5mins': 1, '15mins': 3, '30mins': 6, '1hour': 12, '3hours': 36, '5hours': 60, '1day': 78*1, '3days': 78*3}

        # Compute historical returns for each interval
        for interval_name, interval_bars in intervals.items():
            df[f'hist_ret_{interval_name}'] = df.groupby('ticker')['ret'].apply(
                lambda x: x.iloc[::-1].rolling(window=interval_bars).apply(
                    lambda y: ((y + 1).cumprod()[-1]) - 1, raw=True).iloc[::-1]).shift(interval_bars - 1)

        # Filter the DataFrame to include only the rows for the input date
        df = df[df['date'] == pd.to_datetime(target_date)]

        logging.info(f"Processed data for date: {target_date}")
        return df

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def compute_features(year, input_path, output_path):
    """
    Compute features for all trading days in a given year. Skips processing if the output file already exists.

    Args:
    year (str): The year for which to compute features.
    input_path (str): The directory path containing the trading data files.
    output_path (str): The directory path to save the processed files.

    Returns:
    pd.DataFrame: A DataFrame containing the combined features for all processed days, or None if already processed.
    """
    output_file = os.path.join(output_path, f"{year}_data.parquet")

    # Check if the output file already exists
    if os.path.exists(output_file):
        logging.info(f"Output file {output_file} already exists. Skipping processing.")
        return pd.read_parquet(output_file)

    try:
        # Listing all the relevant files for the given year
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.parquet') and f.startswith(year)]

        # Create a multiprocessing pool to process each file
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(compute_features_single_date, [(f, input_path) for f in files])

        # Combine the results while filtering out None values
        combined_df = pd.concat([df for df in results if df is not None], ignore_index=True)

        # Creating the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Sorting the DataFrame by ticker and date
        combined_df.sort_values(by=['date', 'bar_time', 'ticker'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # Save the combined data to a parquet file
        combined_df.to_parquet(output_file)
        return combined_df

    except Exception as e:
        logging.error(f"Error in compute_features: {e}")
        return None


if __name__ == "__main__":
    # Example usage of the compute_features function
    input_path = '/dat/chbr_group/chbr_scratch/test_mkt_data_labeled'
    output_path = '/dat/chbr_group/chbr_scratch/non_sequential_data'
    YEARS = ['2008', '2009']
    for year in YEARS:
        df = compute_features(year, input_path, output_path)
        logging.info(f'{year} data process completed.')
