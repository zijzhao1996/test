import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
import logging

def calculate_target(args):
    """
    Processes a single file to calculate the target labels.

    Args:
    file_path (str): Path to the input CSV file.
    output_path (str): Path where the processed file will be saved.

    The function reads a CSV file, calculates the target labels,
    and saves the result in the output directory.
    """
    file_path, output_path = args
    output_file = os.path.join(output_path, os.path.basename(file_path)).replace('.csv', '.parquet')

    # Skip processing if the file already exists
    if os.path.exists(output_file):
        logging.info(f"File already processed: {output_file}")
        return

    try:
        df = pd.read_csv(file_path)
        df.sort_values(by=['ticker', 'bar_time'], inplace=True)

        # Calculations for the target
        df['shifted_ret'] = df.groupby('ticker')['ret'].shift(-1)
        df['adj_ret'] = df['shifted_ret'] + 1
        df['target'] = df.groupby('ticker')['adj_ret'].transform(lambda x: x.rolling(6).apply(lambda y: y.prod(), raw=True).shift(-5)) - 1

        df.drop(columns=['shifted_ret', 'adj_ret'], inplace=True)

        df.to_parquet(output_file)
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

def process_files(input_path, output_path):
    """
    Processes multiple files in a directory using multiprocessing.

    Args:
    input_path (str): Directory containing the CSV files to process.
    output_path (str): Directory where processed files will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = [(os.path.join(input_path, f), output_path) for f in os.listdir(input_path) if f.endswith('.csv')]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(calculate_target, files), total=len(files)):
        pass
    pool.close()
    pool.join()

def main(input_path, output_path):
    """
    Main function to execute the label generation process.

    Args:
    input_path (str): Path to the input directory.
    output_path (str): Path to the output directory.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    process_files(input_path, output_path)

if __name__ == "__main__":
    input_path = '/dat/chbr_group/chbr_scratch/test_mkt_data/mktdata'
    output_path = '/data/chbr_group/chbr_scratch/test_mkt_data_labeled/'
    main(input_path, output_path)
