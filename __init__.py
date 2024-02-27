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
from IPython.display import clear_output, Image, display
from sklearn.model_selection import train_test_split

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
get_ipython().run_line_magic('matplotlib', 'inline')

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

class autoencoder():

    def __init__(self, project='wine', dfs=None, latent_size=32, show_models=False):
    
        self.project=project
        self.data=dfs
        
        if not dfs:
            print('Data not found')
        
        if self.project=='optilab':
            Encoder = tf.keras.models.Sequential(name='Encoder')

            Encoder.add(tf.keras.layers.Dense(512,  input_shape = (dfs.shape[1],), activation='relu'))
            Encoder.add(tf.keras.layers.Dense(256,  activation='relu'))
            Encoder.add(tf.keras.layers.Dense(128,  activation='relu'))
            Encoder.add(tf.keras.layers.Dense(64,  activation='relu'))
            Encoder.add(tf.keras.layers.Dense(latent_size,  activation='relu',name='Latent_Space'))
            
            Decoder = tf.keras.models.Sequential(name='Decoder')

            Decoder.add(tf.keras.layers.Dense(64,input_shape = (latent_size,), activation='relu'))
            Decoder.add(tf.keras.layers.Dense(128,  activation='relu'))
            Decoder.add(tf.keras.layers.Dense(256,  activation='relu'))
            Decoder.add(tf.keras.layers.Dense(512,  activation='relu'))
            Decoder.add(tf.keras.layers.Dense(dfs.shape[1],  activation='linear'))
            
            input_layer = tf.keras.layers.Input(shape = (dfs.shape[1]))

            latent_vector = Encoder(input_layer)

            output_layer = Decoder(latent_vector)


            self.autoencoder = tf.keras.models.Model(inputs = input_layer, outputs = output_layer)


            self.autoencoder.compile(tf.keras.optimizers.Adam(learning_rate = 1e-3, clipnorm = 1e-4), loss = "mse") 

            if show_models:
                display(tf.keras.utils.plot_model(self.autoencoder, show_shapes=True))

        elif self.project=='wine':
            
            input_size1 = len(self.data[0].iloc[0])
            input_size2 = len(self.data[1].iloc[0])
            input1 = tf.keras.Input(shape=(input_size1,))
            encoded1 = tf.keras.layers.Dense(512, activation='relu')(input1)
            encoded1 = tf.keras.layers.Dense(512, activation='relu')(encoded1)
            encoded1 = tf.keras.layers.Dense(256, activation='relu')(encoded1)
            encoded1 = tf.keras.layers.Dense(256, activation='relu')(encoded1)

            # Input 2
            input2 = tf.keras.Input(shape=(input_size2,))
            encoded2 = tf.keras.layers.Dense(512, activation='relu')(input2)
            encoded2 = tf.keras.layers.Dense(512, activation='relu')(encoded2)
            encoded2 = tf.keras.layers.Dense(256, activation='relu')(encoded2)
            encoded2 = tf.keras.layers.Dense(256, activation='relu')(encoded2)

            # Merging branches
            merged = tf.keras.layers.concatenate([encoded1, encoded2])
            merged = tf.keras.layers.Dense(512, activation='relu')(merged)
            merged = tf.keras.layers.Dense(256, activation='relu')(merged)
            merged = tf.keras.layers.Dense(128, activation='relu')(merged)

            # Latent space
            latent_space = tf.keras.layers.Dense(latent_size, activation='relu')(merged)  # Adjust the size of the latent space as needed

            # Decoder
            decoded = tf.keras.layers.Dense(128, activation='relu')(latent_space)
            decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
            decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)

            # Splitting branches
            decoded1 = tf.keras.layers.Dense(256, activation='relu')(decoded)
            decoded1 = tf.keras.layers.Dense(256, activation='relu')(decoded1)
            decoded1 = tf.keras.layers.Dense(512, activation='relu')(decoded1)
            decoded1 = tf.keras.layers.Dense(512, activation='relu')(decoded1)
            decoded1 = tf.keras.layers.Dense(input_size1, activation='linear')(decoded1)

            decoded2 = tf.keras.layers.Dense(256, activation='relu')(decoded)
            decoded2 = tf.keras.layers.Dense(256, activation='relu')(decoded2)
            decoded2 = tf.keras.layers.Dense(512, activation='relu')(decoded2)
            decoded2 = tf.keras.layers.Dense(512, activation='relu')(decoded2)
            decoded2 = tf.keras.layers.Dense(input_size2, activation='linear')(decoded2)

            # Autoencoder model
            self.autoencoder = tf.keras.Model(inputs=[input1, input2], outputs=[decoded1, decoded2])

            # Compile the model
            self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3, clipnorm = 1e-4)
                                     , loss='mse')

            # Model summary
            if show_models:
                display(tf.keras.utils.plot_model(self.autoencoder, show_shapes=True))
        else:
            print('Project not supported!!')

    def train(self, batch_size=128, n_epochs=500, patience=50, verbose=1):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        if self.project=='optilab':
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.autoencoder_df, self.file_ids, 
                                                                                    train_size = 0.8, random_state=44)

            self.autoencoder.history = self.autoencoder.fit(x=self.x_train,y=self.x_train,
                                                            validation_data=(self.x_val, self.x_val), 
                                                            batch_size = batch_size, epochs = n_epochs, 
                                                            callbacks = [early_stopping], verbose = verbose).history
        else:
            self.data1_train, self.data1_test, self.data2_train, self.data2_test = train_test_split(self.data[0], self.data[1], test_size=0.2, random_state=44)

            self.data1_train = np.array(list(self.data1_train))
            self.data2_train = np.array(list(self.data2_train))
            self.data1_test = np.array(list(self.data1_test))
            self.data2_test = np.array(list(self.data2_test))

            self.autoencoder.history = self.autoencoder.fit([self.data1_train, self.data2_train],  [self.data1_train, self.data2_train],  
                                                            epochs=n_epochs, batch_size=batch_size,
                                                            callbacks = [early_stopping], verbose = verbose,
                                                            validation_data=([self.data1_test, self.data2_test], [self.data1_test, self.data2_test])).history
            
    def plot_loss(self, grid=True, save=False):

        plt.plot(self.autoencoder.history['loss'], label='Train Loss')
        plt.plot(self.autoencoder.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training History')
        plt.grid(grid)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig('autoencoder_training.png', transparent=True)
        plt.show()

    def plot_prediction_sample(self, grid=True, save=False):
        
        if self.project=='optilab':
            print('Not yet available')
        else:
            sample_index = np.random.randint(1, len(self.data2_test)+1)
            sample = [self.data1_test[sample_index].reshape(1,-1), self.data2_test[sample_index].reshape(1,-1)]
            p1, p2 = self.autoencoder(sample)
            fig, ax = plt.subplots(2, 1, figsize=(15,8), dpi=500)
            ax[0].plot(sample[0].reshape(-1), label='Original')
            ax[0].plot(p1.numpy().reshape(-1), label='Predicted')
            ax[0].grid(grid)
            ax[0].legend()
            ax[1].plot(sample[1].reshape(-1), label='Original')
            ax[1].plot(p2.numpy().reshape(-1), label='Predicted')
            ax[1].grid(grid)
            ax[1].legend()
            plt.suptitle('Sample Prediction')
            plt.tight_layout()
            if save:
                plt.savefig('autoencoder_prediction.png', transparent=True)
            plt.show()

    def save_model(weights_directory = "\\Autoencoder Weights\\", file_name = 'xyz.h5'):

        print('Under Construction')
    
    def load_model(weights_directory = "\\Autoencoder Weights\\", file_name = 'xyz.h5'):

        print('Under Construction')