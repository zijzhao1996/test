# test


import pandas as pd
import os
import multiprocessing
from tqdm import tqdm

def calculate_forward_return(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the dataframe is sorted by ticker and bar_time
        df.sort_values(by=['ticker', 'bar_time'], inplace=True)

        # Calculate forward return for each row
        df['forward_return'] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-6) / x - 1)

        # Save the modified dataframe back to CSV
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    PATH = '/dat/chbr_group/chbr_scratch/test_mkt_data/volumes/'

    # List all CSV files in the directory
    files = [os.path.join(PATH, f) for f in os.listdir(PATH) if f.endswith('.csv')]

    # Use multiprocessing to process files
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(calculate_forward_return, files), total=len(files)):
        pass

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
