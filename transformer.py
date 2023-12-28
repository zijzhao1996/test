import pandas as pd
import os

def obtain_features(path):
    # Read and concatenate the parquet files
    all_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    df_list = [pd.read_parquet(os.path.join(path, file)) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Ensure the dataframe is sorted by ticker and bar_time
    df.sort_values(by=['ticker', 'bar_time'], inplace=True)

    # Define time intervals in terms of 5-minute bars
    intervals = {'5mins': 1, '15mins': 3, '30mins': 6, '1hour': 12, '3hours': 36, 
                 '5hours': 60, '1day': 12*13, '3days': 12*13*3, '5days': 12*13*5, '10days': 12*13*10}

    # Compute historical returns for each interval
    for interval_name, interval_bars in intervals.items():
        # Reverse the dataframe, apply rolling, then reverse it back
        df[f'hist_ret_{interval_name}'] = df.groupby('ticker')['ret'].apply(
            lambda x: x.iloc[::-1].rolling(window=interval_bars, min_periods=1).apply(
                lambda y: (y + 1).prod() - 1, raw=True).iloc[::-1])

    return df

# Example usage
path = '/dat/chbr_group/chbr_scratch/datadump_zijzhao/'
features_df = obtain_features(path)
