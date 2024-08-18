import pandas as pd


# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature engineering
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Week'] = data.index.isocalendar().week
    data['DayOfWeek'] = data.index.dayofweek
    data['Quarter'] = data.index.quarter

    # Add lag features
    for lag in [1, 2, 4, 8, 12]:
        data[f'Sales_Lag_{lag}'] = data['Weekly_Sales'].shift(lag)

    # Add rolling mean features
    for window in [4, 8, 12]:
        data[f'Sales_RollingMean_{window}'] = data['Weekly_Sales'].rolling(window=window).mean()

    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data