# time_series_walmart
The project will use ML/DL to forecast sales and demand based on sample data from Walmart stores while taking into account economic conditions like the Consumer Price Index (CPI), unemployment rate (Unemployment Index, etc). In this project, I want to try a new package called **Optuna**. Optuna is an open-source hyperparameter optimization framework developed by Preferred Networks, Inc. It provides a flexible and efficient platform for optimizing machine learning model hyperparameters, allowing users to find the best set of hyperparameters for their models automatically.

## Environment

- Machine: MacBook with M3 Pro chip (MPS device used for GPU acceleration)
- Python version: 3.9
- Main libraries: PySpark 3.4.1, PyTorch 1.9.0+

## Setup and Installation

1. Clone the repository
   ```
   git clone https://github.com/peeti-sriwongsanguan/time_series_walmart.git
   ```
2. Create a virtual environment & Install the required packages:
   ```
   conda env create -f environment.yml && conda activate walmart-timeseries && conda update -y -n base -c conda-forge conda
   ```

## Requirements

- Python 3.7+
- Java 8 or later
- Torch

## Project Structure

```
walmart-timeseries/
│
├── data/
│   └── walmart_cleaned.csv
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   └── utils.py
│
├── environment.yml
├── main.py
├── requirements.txt
└── README.md
```

4. Make sure your `walmart_cleaned.csv` file is in the `data/` directory.

5. Run the main script:
   ```
   python main.py
   ```