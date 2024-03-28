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

from .models import Autoencoder, ClassificationModel
from .wines import *

# Helper code to make graphs look better
from cycler import cycler
import matplotlib.pyplot as plt
large = 24; medium = 20; small = 16
colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']
params = {'axes.titlesize': small,
          'legend.fontsize': small,
          'figure.figsize': (6, 6),
          'axes.labelsize': small,
          'axes.linewidth': 2,
          'xtick.labelsize': small,
          'xtick.color' : '#1D1717',
          'ytick.color' : '#1D1717',
          'ytick.labelsize': small,
          'axes.edgecolor':'#1D1717',
          'figure.titlesize': medium,
          'axes.prop_cycle': cycler(color = colors),}
plt.rcParams.update(params)

__version__= "1.0"

def get_derivatives(order, wavelengths, values):
    return dif.GLI(order, values, domain_start=min(wavelengths), domain_end=max(wavelengths), num_points=len(wavelengths))

def process_record(record, order):
    wavelengths = record['wavelengths']
    values = record['values']
    derivative = get_derivatives(order, wavelengths, values)
    return {**record, f'derivative_{order}': derivative}

def calculate_derivatives(dataframe, derivative_orders, debug=False):
    records = dataframe.reset_index().to_dict(orient='records')
    tasks = list(product(records, derivative_orders)) 
    if debug:
        print('Starting multiprocessing')
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_record, tasks)

    if debug:
        print('Multiprocessing complete')

    df_results = pd.DataFrame(results)
    
    # Changing 'values' column from list to tuple for grouping
    df_results["values"] = df_results["values"].apply(tuple)
    
    df_results = df_results.groupby(['id', 'values'], as_index=False).first()

    if debug:
        print('Groupby complete')

    if debug:
        return results, df_results.drop(['index'], axis=1)
    return df_results.drop(['index'], axis=1)

def plot_spectra(wavelengths, values, title=None, xlabel=None, ylabel=None, grid=True, save=False):
    
    plt.figure(figsize=(15,6), dpi=500)
    
    plt.plot(wavelengths, values)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Wavelengths')

    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Values')

    plt.grid(grid)
    plt.tight_layout()
    if save:
        plt.savefig('spectra.png', transparent=True) 
    plt.show()

def plot_n_spectra(wavelengths, values, labels='Default',
                   title=None, xlabel=None, ylabel=None, grid=True, legend=True, save=False, ncols=2):
    
    plt.figure(figsize=(15,6), dpi=500)
    
    if labels=='Default':
        labels = np.arange(1, len(wavelengths)+1)

    for i in range(len(wavelengths)):
        plt.plot(wavelengths[i], values[i], label = labels[i])

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Wavelengths')

    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Values')

    if legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncols)

    plt.grid(grid)
    plt.tight_layout()
    if save:
        plt.savefig('spectra.png', transparent=True) 
    plt.show()