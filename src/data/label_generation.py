import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
import logging

def calculate_target(file_path, output_path):
    """
    Processes a single file to calculate the target labels.

    Args:
    file_path (str): Path to the input CSV file.
    output_path (str): Path where the processed file will be saved.

    The function reads a CSV file, calculates the target labels,
    and saves the result in the output directory.
    """
    try:
        df = pd.read_csv(file_path)
        df.sort_values(by=['ticker', 'bar_time'], inplace=True)

        # Calculate the target labels with a specified logic
        df['target'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(-6) / x - 1)

        output_file = os.path.join(output_path, os.path.basename(file_path))
        df.to_csv(output_file, index=False)
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

    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(lambda f: calculate_target(f, output_path), files), total=len(files)):
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_files(input_path, output_path)

if __name__ == "__main__":
    input_path = 'path_to_input_directory'
    output_path = 'path_to_output_directory'
    main(input_path, output_path)
