import os
import numpy as np
import pandas as pd

from glob import glob
from tqdm.auto import tqdm
from prettytable import PrettyTable
from IPython.display import clear_output
from pandas.errors import ParserError

from .models import Autoencoder, ClassificationModel

def convert_to_float(value):
    return float(value.replace(';', '.'))

def input_files(data_folder, save_folder="Input"):
    
    wine = os.path.join(data_folder, 'WineScan')
    photometry = os.path.join(data_folder, 'Espectrofotometro')
    years = [i.split('\\')[-1] for i in glob(wine+'\*')]
    winedf = pd.DataFrame(columns=['id', 'wavelengths', 'values'])
    photometrydf = pd.DataFrame(columns=['id', 'wavelengths', 'values'])
    for y in years:
        winetemp = glob(os.path.join(wine, y)+'\*.csv')
        photometrytemp = glob(os.path.join(photometry, y)+'\*.csv')
        
        for i in winetemp:
            df = pd.read_csv(i, delimiter=';', decimal=',', skiprows=6, encoding='latin-1', header=None)
            id = i.split('\\')[-1].split('.')[0]
            
            try:
                w = list(map(float, df[:1060][0].values))
            except Exception as e:
                print(f'Id: {id}')
                print(e)
                continue
            v = []
            for k in df.columns:
                if k==0:
                    pass
                else:
                    try:
                        v.append(list(map(float, df[:1060][k].values)))
                        v.append(list(map(float, df[1074:][k].values)))
                    except Exception as e:
                        print(f'Id: {id}')
                        print(e)
                        continue           

            for j in range(len(v)):
                record = {'id': id,
                        'wavelengths':w,
                        'values':v[j]}
                winedf = winedf.append(record, ignore_index=True)
                
        for i in photometrytemp:
            if y=='2021' or y=='2020':
                try: 
                    df = pd.read_csv(i, skiprows=7, encoding='latin-1', header=None)
                except ParserError as e:
                    print(i)
                except Exception as e:
                    print(e)
            else:
                try: 
                    df = pd.read_csv(i, skiprows=19, encoding='latin-1', header=None)
                except ParserError as e:
                    print(i)
                except Exception as e:
                    print(e)
            df.dropna(axis=1, inplace=True)
            try:
                df = df.replace('****', np.nan).replace(';', np.nan).interpolate(method='linear')
            except Exception as e:
                df = pd.read_csv(i, delimiter=';', decimal=',',skiprows=6, encoding='latin-1', header=None)
                df = df.replace('****', np.nan).replace(';', np.nan).interpolate(method='linear')
                
            try:
                df = df.astype(float)
            except:
                print(i)
            id = i.split('\\')[-1].split('.')[0].split('-')[0]
            try:
                w = list(map(float, df[141:][0].values))
            except: 
                try:
                    w = list(map(convert_to_float, df[141:][0].values))
                except Exception as e:
                    print(f'Id: {id}')
                    print(e)
                    continue
            v = []
            for k in df.columns:
                if k==0:
                    pass
                else:
                    try:
                        v.append(pd.Series(list(map(float, df[141:][k].values))).interpolate().values)
                    except Exception as e:
                        print(f'Id: {id}')
                        print(e)
                        continue
            for j in range(len(v)):
                if len(v[j])==770:
                    record = {'id': id,
                            'wavelengths':w,
                            'values':v[j]}
                    photometrydf = photometrydf.append(record, ignore_index=True)

    finaldf = pd.merge(winedf, photometrydf, on='id', how='inner', suffixes=('_wine', '_photometry'))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    winedata = os.path.join(save_folder, 'Wines')
    photometrydata = os.path.join(save_folder, 'Photometry')
    if not os.path.exists(winedata):
        os.makedirs(winedata)
    if not os.path.exists(photometrydata):
        os.makedirs(photometrydata)

    n = len(glob(save_folder + "\*.pickle")) + 1

    filename = os.path.join(save_folder, f'data_{n}.pickle')
    winename =  os.path.join(winedata, f'data_{n}.pickle')
    photometryname =  os.path.join(photometrydata, f'data_{n}.pickle')

    finaldf.to_pickle(filename)
    winedf.to_pickle(winename)
    photometrydf.to_pickle(photometryname)

    return finaldf, winedf, photometrydf

def input_2020_files(data_folder, save_folder="Input"):
    pass

def read_input_file(folder="input", file='', verbose=0):
    if verbose==1:
        data=pd.read_pickle(os.path.join(folder,file))
        winedata=pd.read_pickle(os.path.join(folder, "Wines", file))
        photometrydata=pd.read_pickle(os.path.join(folder, "Photometry", file))
        return data, winedata, photometrydata
    else: 
        data=pd.read_pickle(os.path.join(folder,file))
        return data

def read_input_files(folder="Input", verbose=0):

    
    if verbose==1:
        data_files = glob(folder + "\*.pickle")
        wine_files = glob(os.path.join(folder, 'Wines') + "\*.pickle")
        photometry_files = glob(os.path.join(folder, 'Photometry') + "\*.pickle")
        if len(data_files)==1:
            data=pd.read_pickle(data_files[0])
            winedata=pd.read_pickle(wine_files[0])
            photometrydata=pd.read_pickle(photometry_files[0])
        else:
            dfs = {}
            for i in range(len(data_files)):
                dfs[i] = pd.read_pickle(data_files[i])

            data = pd.concat(list(dfs.values()))
            wdfs = {}
            for i in range(len(wine_files)):
                wdfs[i] = pd.read_pickle(wine_files[i])

            winedata = pd.concat(list(wdfs.values()))
            pdfs = {}
            for i in range(len(photometry_files)):
                pdfs[i] = pd.read_pickle(photometry_files[i])

            photometrydata = pd.concat(list(pdfs.values()))
        return data, winedata, photometrydata
    else:
        data_files = glob(folder + "\*.pickle")
        if len(data_files)==1:
            data=pd.read_pickle(data_files[0])
        else:
            dfs = {}
            for i in range(len(data_files)):
                dfs[i] = pd.read_pickle(data_files[i])

            data = pd.concat(list(dfs.values()))
        return data
    
def merge_input_result(results_file, input_file, folder="Input"):
    
    """
    Merges the results from a results file with the corresponding input data from a separate file.

    Parameters:
    results_file (str): The filename of the results file containing the ID and class information.
    input_file (str): The filename of the input data file containing wavelenghts and values corresponding to the IDs.
    folder (str, optional): The folder directory where the input and results files are located. Default is "Input".

    Returns:
    merged_df (DataFrame): A pandas DataFrame containing the merged data, with the results and input data combined.
    """
    
    merged_df = pd.read_pickle(os.path.join(folder,input_file))
    results_df = pd.read_csv(os.path.join(folder,results_file))
    merged_df = pd.merge(results_df, merged_df, right_on='id', left_on='ID', how='inner').drop(["ID"], axis=1)
    # classes = merged_df["class"]
    # merged_df.drop(["class"],axis=1,inplace=True)
    return merged_df
        
    
#TODO: train_models and infer function needs to be fixed according to the input that we are taking.
# def train_models(data_folder):

#     data = read_input_files(data_folder)
    
#     try:
#         autoencoder = Autoencoder(latent_size=32, show_models=True) 
#     except:
#         autoencoder = Autoencoder(latent_size=32, show_models=False) 
#     autoencoder.train(data) 

#     encoder = autoencoder.get_encoder(trained=True)
#     X_encoded = encoder.predict(data)  
    

#     classification_model = ClassificationModel(encoder, latent_size=32, num_classes=4)
#     classification_model.train(X_encoded, y)

#     autoencoder.save_model('autoencoder_model.h5')
#     classification_model.save_weights('classification_model_weights.h5')

# def infer(input_data):
#     autoencoder = Autoencoder.load_model('autoencoder_model.h5')
#     classification_model = ClassificationModel.load_weights('classification_model_weights.h5', encoder=autoencoder.get_encoder())
    
#     encoded_input = autoencoder.get_encoder().predict(input_data)
    
#     prediction = classification_model.predict(encoded_input)

#     return prediction