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

class autoencoder():

    def __init__(self, project='wine', dfs=None, latent_size=32, show_models=False):
    
        self.project=project
        self.data=dfs
        self.trained=0
        
        samples_count = [len(df) for df in self.data]
        if len(set(samples_count)) == 1:
            print("All DataFrames have the same number of rows.")
        else:
            raise ValueError("DataFrames do not have the same number of samples.")
        
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
            
            input_layers = []
            input_sizes = []
            encoded_layers = []
            decoded_outputs = []
            
            for df in self.data:
                for col in df.columns:
                    input_size = len(df[col].iloc[0])
                    input_sizes.append(input_size)
                    input_layer = tf.keras.Input(shape=(input_size,))
                    input_layers.append(input_layer)
                    
                    encoded = tf.keras.layers.Dense(512, activation='relu')(input_layer)
                    encoded = tf.keras.layers.Dense(512, activation='relu')(encoded)
                    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
                    encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
                    encoded_layers.append(encoded)
            
            merged = tf.keras.layers.concatenate(encoded_layers)
            merged = tf.keras.layers.Dense(512, activation='relu')(merged)
            merged = tf.keras.layers.Dense(256, activation='relu')(merged)
            merged = tf.keras.layers.Dense(128, activation='relu')(merged)
            
            latent_space = tf.keras.layers.Dense(latent_size, activation='relu', name="Latent_Space")(merged)
            
            decoded = tf.keras.layers.Dense(128, activation='relu')(latent_space)
            decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
            decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
            
            for input_size in input_sizes:
                
                decoded_ = tf.keras.layers.Dense(256, activation='relu')(decoded)
                decoded_ = tf.keras.layers.Dense(256, activation='relu')(decoded_)
                decoded_ = tf.keras.layers.Dense(512, activation='relu')(decoded_)
                decoded_ = tf.keras.layers.Dense(512, activation='relu')(decoded_)
                decoded_ = tf.keras.layers.Dense(input_size, activation='linear')(decoded_)
                decoded_outputs.append(decoded_)
                
            self.autoencoder = tf.keras.Model(inputs=input_layers, outputs=decoded_outputs)

            self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3, clipnorm = 1e-4)
                                     , loss='mse')
            
            self.encoder = tf.keras.Model(inputs=input_layers, outputs=latent_space)
            
            if show_models:
                display(tf.keras.utils.plot_model(self.autoencoder, show_shapes=True))
        else:
            print('Project not supported!!')
            
    def get_encoder(self):
        if self.encoder is not None:
            if self.trained:
                return self.encoder
            else:
                self.train()
                return self.encoder
        else:
            print("Encoder Model is not defined.")
            return None
                
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
            self.X_train_list = []
            self.X_val_list = []

            for df in self.data:

                for col in df.columns:
                    X = np.array(list(df[col]))
                    
                    X_train, X_val = train_test_split(X, test_size=0.2, random_state=44)
                    
                    self.X_train_list.append(X_train)
                    self.X_val_list.append(X_val)
                    
            self.autoencoder.history = self.autoencoder.fit(
                                                            x=self.X_train_list,
                                                            y=self.X_train_list,
                                                            epochs=n_epochs,
                                                            batch_size=batch_size,
                                                            callbacks=[early_stopping],
                                                            verbose=verbose,
                                                            validation_data=(self.X_val_list, self.X_val_list)
                                                        ).history
            
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
            num_inputs = len(self.X_val_list)
            sample_indices = np.random.randint(0, len(self.X_val_list[0]), 1)
            
            samples = [self.X_val_list[i][sample_indices].reshape(1, -1) for i in range(num_inputs)]
            predictions = self.autoencoder.predict(samples)
            
            fig, axs = plt.subplots(num_inputs, 1, figsize=(15, 8 * num_inputs), dpi=100)
            for i in range(num_inputs):
                original = samples[i].flatten()
                predicted = predictions[i].flatten()
                
                if num_inputs > 1:
                    ax = axs[i]
                else:
                    ax = axs
                    
                ax.plot(original, label='Original')
                ax.plot(predicted, label='Predicted')
                ax.grid(grid)
                ax.legend()
                ax.set_title(f'Sample Prediction for Input {i+1}')
            
            plt.suptitle('Autoencoder Sample Predictions')
            plt.tight_layout()
            
            if save:
                plt.savefig('autoencoder_prediction.png', transparent=True)
            
            plt.show()
            
            # sample_index = np.random.randint(1, len(self.data2_test)+1)
            # sample = [self.data1_test[sample_index].reshape(1,-1), self.data2_test[sample_index].reshape(1,-1)]
            # p1, p2 = self.autoencoder(sample)
            # fig, ax = plt.subplots(2, 1, figsize=(15,8), dpi=500)
            # ax[0].plot(sample[0].reshape(-1), label='Original')
            # ax[0].plot(p1.numpy().reshape(-1), label='Predicted')
            # ax[0].grid(grid)
            # ax[0].legend()
            # ax[1].plot(sample[1].reshape(-1), label='Original')
            # ax[1].plot(p2.numpy().reshape(-1), label='Predicted')
            # ax[1].grid(grid)
            # ax[1].legend()
            # plt.suptitle('Sample Prediction')
            # plt.tight_layout()
            # if save:
            #     plt.savefig('autoencoder_prediction.png', transparent=True)
            # plt.show()

    def save_model(self, weights_directory = "Autoencoder Weights", file_name = 'xyz.h5'):

        if not os.path.exists(weights_directory):
            os.makedirs(weights_directory)
        file=os.path.join(weights_directory, file_name)
        self.autoencoder.save(file)
    
    def load_model(self, model_weights):
        self.trained=1
        self.autoencoder=tf.keras.models.load_model(model_weights)