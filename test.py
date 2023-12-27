import pandas as pd
import os
import multiprocessing
from tqdm import tqdm
import time

def calculate_forward_return(args):
    file_path, output_path = args
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the dataframe is sorted by ticker and bar_time
        df.sort_values(by=['ticker', 'bar_time'], inplace=True)

        # Shift 'ret', adjust it, and compute the forward return
        df['adj_ret'] = df.groupby('ticker')['ret'].shift(-1) + 1
        df['target'] = df.groupby('ticker')['adj_ret'].transform(
            lambda x: x.rolling(6, min_periods=1).apply(lambda y: y.prod(), raw=True).shift(-6)) - 1

        # Remove the 'adj_ret' column as it's no longer needed
        df.drop(columns=['adj_ret'], inplace=True)

        # Determine the output file path
        output_file = os.path.join(output_path, os.path.basename(file_path))

        # Save the modified dataframe
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    start_time = time.time()  # Start timing

    input_path = '/dat/chbr_group/chbr_scratch/test_mkt_data/volumes/'
    output_path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all CSV files in the input directory
    files = [(os.path.join(input_path, f), output_path) for f in os.listdir(input_path) if f.endswith('.csv')]

    # Use multiprocessing to process files
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(calculate_forward_return, files), total=len(files)):
        pass

    pool.close()
    pool.join()

    end_time = time.time()  # End timing
    print(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
