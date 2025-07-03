try:
    import kaggle
except ImportError:
    raise ImportError("Please install the Kaggle API client: pip install kaggle")

import pandas as pd


def write_filtered_data(df: pd.DataFrame, start_date = '2023-01-01', end_date = '2025-07-03 00:48:00') -> pd.DataFrame:
    # Ensure index is in datetime format
    index = pd.to_datetime(df['Timestamp'], unit='s')
    
    # Filter for dates between start_date and end_date
    df = df[(index >= pd.Timestamp(start_date)) & (index <= pd.Timestamp(end_date))]

    # Return only the "Open" column
    df = df[['Timestamp', 'Close']]

    df.to_csv("./applications/market_making/data/btcusd_1-min_data_filtered.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("./applications/market_making/data/btcusd_1-min_data.csv")
    write_filtered_data(df)
    print(f"Feel free to delete the original file: ./applications/market_making/data/btcusd_1-min_data.csv")
