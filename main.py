from src.data_preprocessing import load_and_preprocess_data
import time
import logging
import pandas as pd
pd.set_option('display.max_columns', None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Starting the prediction process...")
    zip_filepath = r'data/'
    csv_filename = 'walmart_cleaned.csv'
    # Load and preprocess data
    data = load_and_preprocess_data(f'{zip_filepath}{csv_filename}')

    return data

if __name__ == "__main__":
    df = main()
    print(df.head())
