import os
import json
import time
import shutil
import datetime
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tqdm.auto import tqdm
import multiprocessing
from itertools import product
import differint.differint as dif
from prettytable import PrettyTable
from joblib import Parallel, delayed
from IPython.display import clear_output

__version__= "1.0"

def get_derivatives(order, wavelengths, values):
    return dif.GLI(order, values, domain_start=min(wavelengths), domain_end=max(wavelengths), num_points=len(wavelengths))

def process_record(record, order):
    wavelengths = record['wavelengths']
    values = record['values']
    derivative = get_derivatives(order, wavelengths, values)
    return {**record, f'derivative_{order}': derivative}

def calculate_derivatives(dataframe, derivative_orders):
    records = dataframe.reset_index().to_dict(orient='records')
    tasks = list(product(records, derivative_orders))  # This creates an iterable of tuples

    # # Optional: Print a few tasks to verify their structure
    # print("Sample tasks:", tasks[:5])

    with multiprocessing.Pool() as pool:
        # Map tasks to the process_record function
        results = pool.starmap(process_record, tasks)

    # Convert list of dictionaries to DataFrame
    df_results = pd.DataFrame(results)

    # Group by the index column
    index_col = 'index'  # This is the name of the column that holds the original index
    grouped = df_results.groupby(index_col)

    # Combine derivatives of the same record into one row
    reshaped_df = grouped.apply(lambda x: x.iloc[0]).reset_index(drop=True)

    for order in derivative_orders:
        reshaped_df[f'derivative_{order}'] = grouped.apply(lambda x: x[f'derivative_{order}'].iloc[0]).values

    return reshaped_df.drop(['index'], axis=1)