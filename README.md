# test

import pandas as pd
import os
import multiprocessing
from tqdm import tqdm

def calculate_forward_return(file_path, output_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the dataframe is sorted by ticker and bar_time
        df.sort_values(by=['ticker', 'bar_time'], inplace=True)

        # Calculate forward return for each row
        df['target'] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-6) / x - 1)

        # Determine the output file path
        output_file = os.path.join(output_path, os.path.basename(file_path))

        # Save the modified dataframe
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    input_path = '/dat/chbr_group/chbr_scratch/test_mkt_data/volumes/'
    output_path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # List all CSV files in the input directory
    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]

    # Use multiprocessing to process files
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(lambda f: calculate_forward_return(f, output_path), files), total=len(files)):
        pass

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
