import os
from datetime import datetime

path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'

# Extract dates from file names
dates = []
for file in os.listdir(path):
    if file.endswith('.parquet'):
        date_str = file.split('.')[0]
        dates.append(datetime.strptime(date_str, '%Y%m%d'))

# Sort the dates
dates.sort()


# Save the sorted dates to a file
with open('sorted_dates.txt', 'w') as f:
    for date in dates:
        f.write(date.strftime('%Y%m%d') + '\n')


def get_previous_n_days(target_date, n):
    # Read the sorted dates
    with open('sorted_dates.txt', 'r') as f:
        sorted_dates = [datetime.strptime(line.strip(), '%Y%m%d') for line in f]

    # Convert target_date to datetime
    target_date = datetime.strptime(target_date, '%Y%m%d')

    # Find the target date in the list
    if target_date in sorted_dates:
        target_index = sorted_dates.index(target_date)
        # Get previous N days
        start_index = max(0, target_index - n)
        return [date.strftime('%Y%m%d') for date in sorted_dates[start_index:target_index]]
    else:
        return []

# Example usage
previous_days = get_previous_n_days('20090605', 5)
print(previous_days)
