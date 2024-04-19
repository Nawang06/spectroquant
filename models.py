import os
import numpy as np
import tensorflow as tf
import keras.backend as K

from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, LeakyReLU,MaxPool1D,UpSampling1D, Lambda, GlobalMaxPool1D,Concatenate, concatenate, ZeroPadding1D, Cropping1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix


from IPython.display import display

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

class Autoencoder():

    def __init__(self, project='wine', dfs=None, latent_size=32, show_models=False):
    
        self.project=project
        self.data=dfs
        self.loaded=False
        self.trained=False
        self.column_names = []
        self.file_ids = []
        
        samples_count = [len(df) for df in self.data]
        if len(set(samples_count)) != 1:
            raise ValueError("DataFrames do not have the same number of samples.")
            
        if not dfs:
            print('Data not found')
        
        self._build_model(latent_size, show_models)
        
                    
    def _build_model(self, latent_size, show_models):
        
        if self.project=='optilab':
            Encoder = tf.keras.models.Sequential(name='Encoder')

            Encoder.add(tf.keras.layers.Dense(512,  input_shape = (self.data.shape[1],), activation='relu'))
            Encoder.add(tf.keras.layers.Dense(256,  activation='relu'))
            Encoder.add(tf.keras.layers.Dense(128,  activation='relu'))
            Encoder.add(tf.keras.layers.Dense(64,  activation='relu'))
            Encoder.add(tf.keras.layers.Dense(latent_size,  activation='relu',name='Latent_Space'))
            
            Decoder = tf.keras.models.Sequential(name='Decoder')

            Decoder.add(tf.keras.layers.Dense(64,input_shape = (latent_size,), activation='relu'))
            Decoder.add(tf.keras.layers.Dense(128,  activation='relu'))
            Decoder.add(tf.keras.layers.Dense(256,  activation='relu'))
            Decoder.add(tf.keras.layers.Dense(512,  activation='relu'))
            Decoder.add(tf.keras.layers.Dense(self.data.shape[1],  activation='linear'))
            
            input_layer = tf.keras.layers.Input(shape = (self.data.shape[1]))

            latent_vector = Encoder(input_layer)

            output_layer = Decoder(latent_vector)

            self.autoencoder = tf.keras.models.Model(inputs = input_layer, outputs = output_layer)


            self.autoencoder.compile(tf.keras.optimizers.Adam(learning_rate = 1e-3, clipnorm = 1e-4), loss = "mse") 

            if show_models:
                display(tf.keras.utils.plot_model(self.autoencoder, show_shapes=True))

        elif self.project=='wine':
            decoded_outputs = []
            
            photometry_inputs = []
            photometry_encoded = []
            winescan_inputs = []
            winescan_encoded = []
            
            def enc_block(x, nch=32, fs=7):
                x = Conv1D( nch, fs, padding='same' )( x )
                x = LeakyReLU()( x )
                x = MaxPool1D()( x )
                return x
            
            def dec_block( x, nch=32, fs=7):
                x = Conv1D( nch, fs, padding='same' )( x )
                x = LeakyReLU()( x )
                x = UpSampling1D()( x )
                return x

            for df in self.data:
                self.file_ids = df["id"].to_list()  # Assuming each df has an 'id' column

            for i in df.columns:
                if 'photometry' in i:
                    size_photometry = np.array(df[i][0]).shape[0]
                if 'wines' in i:
                    size_wines = np.array(df[i][0]).shape[0]
            
            final_size = max(size_photometry, size_wines)
            
            for col in df.columns:
                if "id" in col.lower() or "wavelength" in col.lower():
                    continue
                self.column_names.append(col)
                input_size = len(df[col].iloc[0])
                input_layer = tf.keras.Input(shape=(input_size,1), name=col)
                if "photometry" in col.lower():
                    padding_layer = tf.keras.layers.ZeroPadding1D(padding= (0, final_size-size_photometry))(input_layer)
                    x1 = enc_block( padding_layer ) # (/2) 512, 10
                    x2 = enc_block( x1 ) # (/4) 256, 10
                    x3 = enc_block( x2 ) # (/8) 128, 10
                    encoded = enc_block( x3 ) # (/16) 64, 10
                    photometry_inputs.append(input_layer)
                    photometry_encoded.append(encoded)
                elif "winescan" in col.lower():
                    x1 = enc_block( input_layer ) # (/2) 512, 10
                    x2 = enc_block( x1 ) # (/4) 256, 10
                    x3 = enc_block( x2 ) # (/8) 128, 10
                    encoded = enc_block( x3 ) # (/16) 64, 10
                    winescan_inputs.append(input_layer)
                    winescan_encoded.append(encoded)
            
                # Concatenate and add category-specific dense layers
                if photometry_encoded:
                    photometry_merged = tf.keras.layers.concatenate(photometry_encoded)
                    # Additional dense layers for Photometry branch
                    photometry_merged = enc_block(photometry_merged)
                    photometry_merged = enc_block(photometry_merged)
                else:
                    photometry_merged = None

                if winescan_encoded:
                    winescan_merged = tf.keras.layers.concatenate(winescan_encoded)
                    # Additional dense layers for Winescan branch
                    winescan_merged = enc_block(winescan_merged)
                    winescan_merged = enc_block(winescan_merged)
                else:
                    winescan_merged = None
            
            merged = tf.keras.layers.concatenate([photometry_merged, winescan_merged])
            
            # Continue with shared layers
            x5 = GlobalMaxPool1D()( merged )
            xc = Dense( latent_size )( x5 )
            xc = LeakyReLU()( xc )
            
            xcu = Dense( int(input_size/16) )( xc )
            xcu = LeakyReLU()( xcu )
            xcu = Lambda( lambda x: tf.expand_dims( x, -1 ) )( xcu )
                
            # Decoder
            decoded_outputs = []
            
            # Decoder
            for input_layer in photometry_inputs:
                x6 = dec_block(xcu) # 64, 10
                x7 = dec_block(x6) # 128, 10
                x8 = dec_block(x7) # 256, 10
                x9 = dec_block(x8) # 512, 10
                xout = GlobalMaxPool1D()( x9 )
                xout = Dense(input_layer.shape[1], activation="linear")(xout)
                decoded_outputs.append(xout)

            for input_layer in winescan_inputs:
                x6 = dec_block(xcu)
                x7 = dec_block(x6) 
                x8 = dec_block(x7) 
                x9 = dec_block(x8) 
                xout = GlobalMaxPool1D()( x9 )
                xout = Dense(input_layer.shape[1],activation="linear")(xout)
                decoded_outputs.append(xout)
            
            self.autoencoder = tf.keras.Model(inputs=photometry_inputs + winescan_inputs, outputs=decoded_outputs)
            self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1e-4), loss='mse')
            self.encoder = tf.keras.Model(inputs=photometry_inputs + winescan_inputs, outputs=xc)            
            if show_models:
                display(tf.keras.utils.plot_model(self.autoencoder, show_shapes=True))
        else:
            print('Project not supported!!')

        
            
    def get_encoder(self, trained=False):
        if trained:
            if self.loaded or self.trained:
                return self.encoder
            else:
                self.train()
                return self.encoder
        else:
            return self.encoder
        
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
                    if "wavelength" in col.lower() or "id" in col.lower():
                        continue
                    else:
                        X = np.array(list(df[col]))
                        
                        X_train, X_val = train_test_split(X, test_size=0.2, random_state=44)
                        
                        self.X_train_list.append(X_train)
                        self.X_val_list.append(X_val)
                    
            self.id_trains, self.id_vals = train_test_split(self.file_ids, test_size=0.2,random_state=44)
                    
            self.autoencoder.history = self.autoencoder.fit(
                                                            x=self.X_train_list,
                                                            y=self.X_train_list,
                                                            epochs=n_epochs,
                                                            batch_size=batch_size,
                                                            callbacks=[early_stopping],
                                                            verbose=verbose,
                                                            validation_data=(self.X_val_list, self.X_val_list)
                                                        ).history
        self.trained=True
            
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
            sample_index = np.random.randint(0, len(self.X_val_list[0]), 1)[0]
            
            samples = [self.X_val_list[i][sample_index].reshape(1, -1) for i in range(num_inputs)]
            predictions = self.autoencoder.predict(samples)
            
            column_names = self.column_names 
            
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
                ax.set_title(f'Sample Prediction for Input {column_names[i]}')
            
            plt.suptitle(f'Autoencoder Sample Predictions for id: {self.id_vals[sample_index]}')
            plt.tight_layout(pad=3.0)
            
            if save:
                plt.savefig('autoencoder_prediction.png', transparent=True)
            
            plt.show()

    def save_model(self, weights_directory = "Autoencoder Weights", file_name = 'xyz.h5'):
        if not os.path.exists(weights_directory):
            os.makedirs(weights_directory)
        file=os.path.join(weights_directory, file_name)
        self.autoencoder.save(file)
    
    def load_model(self, model_weights):
        self.loaded=True
        self.autoencoder=tf.keras.models.load_model(model_weights)

class ClassificationModel:
    
    def __init__(self, encoder, latent_size=32, num_classes=4):
        self.encoder = encoder
        self.latent_size = latent_size
        self.num_classes = num_classes
        self._build_model()
    
    def _build_model(self):
        
        encoded_input = Input(shape=(self.latent_size, 1))
        
        x = tf.keras.layers.Conv1D(32, 5, activation='relu',padding="same")(encoded_input)
        x = tf.keras.layers.Conv1D(32, 5, activation='relu',padding="same")(x)
        x = tf.keras.layers.Conv1D(32, 5, activation='relu',padding="same")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(64, 5, activation='relu',padding="same")(x)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu',padding="same")(x)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu',padding="same")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding="same")(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding="same")(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding="same")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(128, 3, activation='relu',padding="same")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

        x = tf.keras.layers.Flatten()(x)
        xx = tf.keras.layers.Dense(128, activation='relu')(x)
        xx = tf.keras.layers.Dense(128, activation='relu')(xx)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(xx)

        self.classifier = Model(inputs=encoded_input, outputs=output)

    
    def _preprocess_data(self, df, trainset=False):
        
        processed_data = []
        labels = df["class"]
        data = df.drop("class", axis=1)
        if trainset:
            labels = self.lb.fit_transform(labels) 
        else:
            labels = self.lb.transform(labels) 
                
        for column in data.columns:
            if "wavelength" in column.lower() or "id" in column.lower():
                continue
            # Convert tuples to numpy array and store in the list
            column_data = np.array(data[column].tolist())
            processed_data.append(column_data)
            
        encoded_input = self.encoder.predict(processed_data)
        
        return encoded_input, labels
        

    def train(self, df,test_df=None, batch_size=64, epochs=500, patience=20,test_ratio=0.2, valid_ratio=0.2):
        
        self.lb = LabelBinarizer()
        if test_df is not None:
            ids_train, ids_valid = train_test_split(df['id'].unique(), test_size = valid_ratio)
            self.df_train = df[df['id'].isin(ids_train)]
            self.df_valid = df[df['id'].isin(ids_valid)]            
            self.train_input, self.train_labels = self._preprocess_data(self.df_train, trainset=True)
            self.test_input, self.test_labels = self._preprocess_data(test_df)
            self.val_input, self.val_labels = self._preprocess_data(self.df_valid)
        else:
            ids_train, ids_test = train_test_split(df['id'].unique(), test_size = test_ratio)
            ids_train, ids_valid = train_test_split(ids_train, test_size = valid_ratio)
            
            self.df_train = df[df['id'].isin(ids_train)]
            self.df_valid = df[df['id'].isin(ids_valid)]
            self.df_test = df[df['id'].isin(ids_test)]
            
            self.train_input, self.train_labels = self._preprocess_data(self.df_train, trainset=True)
            self.test_input, self.test_labels = self._preprocess_data(self.df_test)
            self.val_input, self.val_labels = self._preprocess_data(self.df_valid)
            
        
        optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.classifier.compile(optimizer=optim, loss='categorical_crossentropy',metrics=["accuracy"])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        self.classifier.history = self.classifier.fit(self.train_input, self.train_labels, validation_data=(self.val_input, self.val_labels), 
                                 epochs=epochs, batch_size=batch_size, callbacks=[early_stopping]).history
        
    
    def plot_loss(self, grid=True, save=False):
        plt.plot(self.classifier.history['loss'], label='Train Loss')
        plt.plot(self.classifier.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Classifier Training History')
        plt.grid(grid)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig('classifier.png', transparent=True)
        plt.show()
        

    def save_model(self, weights_directory = "Classifier Weights", file_name = 'classifier.h5'):
        if not os.path.exists(weights_directory):
            os.makedirs(weights_directory)
        file=os.path.join(weights_directory, file_name)
        self.autoencoder.save(file)
    
    def load_model(self, model_weights):
        self.loaded=True
        self.autoencoder=tf.keras.models.load_model(model_weights)