import os
from datetime import datetime

def get_previous_n_days(target_date, n, data_path, dates_file='sorted_dates.txt'):
    """
    Retrieves the previous N days for a given target date. It extracts and sorts 
    trading dates from file names if not already done.

    Args:
    target_date (str): The target date in '%Y%m%d' format.
    n (int): The number of days to retrieve.
    data_path (str): The directory path containing the trading data files.
    dates_file (str): The file to read or save sorted dates.

    Returns:
    List[str]: Dates in '%Y%m%d' format for the target date and the previous N days.
    """

    # Check if dates are already extracted and sorted
    if not os.path.exists(dates_file):
        dates = []
        for file in os.listdir(data_path):
            if file.endswith('.parquet'):
                date_str = file.split('.')[0]
                try:
                    dates.append(datetime.strptime(date_str, '%Y%m%d'))
                except ValueError:
                    continue  # Skip files that don't match the expected format

        dates.sort()

        # Save the sorted dates to a file
        with open(dates_file, 'w') as f:
            for date in dates:
                f.write(date.strftime('%Y%m%d') + '\n')

    # Read the sorted dates
    with open(dates_file, 'r') as f:
        sorted_dates = [datetime.strptime(line.strip(), '%Y%m%d') for line in f]

    # Convert target_date to datetime and find the previous N days
    target_date = datetime.strptime(target_date, '%Y%m%d')
    if target_date in sorted_dates:
        target_index = sorted_dates.index(target_date)
        start_index = max(0, target_index - n)
        return [date.strftime('%Y%m%d') for date in sorted_dates[start_index:target_index + 1]]
    else:
        return []

# Example usage
data_path = '/dat/chbr_group/chbr_scratch/test_mkt_data_labeled/'
previous_days = get_previous_n_days('20090605', 5, data_path)
print(previous_days)
