import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
import time

def process_file(args):
    file_name, market_data_path, volume_data_path, output_path = args
    try:
        # Read market and volume data CSV files
        market_df = pd.read_csv(os.path.join(market_data_path, file_name))
        volume_df = pd.read_csv(os.path.join(volume_data_path, file_name))

        # Merge dataframes on 'date', 'bar_time', 'ticker'
        merged_df = pd.merge(market_df, volume_df, on=['date', 'bar_time', 'ticker'])

        # Save the merged dataframe as a parquet file
        merged_df.to_parquet(os.path.join(output_path, file_name.replace('.csv', '.parquet')))
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

def main():
    start_time = time.time()  # Start timing

    market_data_path = '/dat/chbr_group/chbr_scratch/test_mkt_data/mktdata/'
    volume_data_path = '/dat/chbr_group/chbr_scratch/test_mkt_data/volumes/'
    output_path = '/dat/chbr_group/chbr_scratch/datadump_merged/'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all CSV files in the market data and volume data directories
    market_files = set(os.listdir(market_data_path))
    volume_files = set(os.listdir(volume_data_path))
    
    # Process only the files that are present in both directories
    common_files = market_files.intersection(volume_files)

    # Create a list of arguments for each file to be processed
    file_args = [(f, market_data_path, volume_data_path, output_path) for f in common_files]

    # Use multiprocessing to process files
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(process_file, file_args), total=len(common_files)):
        pass

    pool.close()
    pool.join()

    end_time = time.time()  # End timing
    print(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
