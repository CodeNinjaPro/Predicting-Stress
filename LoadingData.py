# importing libraries 
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import multiprocessing
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt, find_peaks

# making variable for data paths
DATA_PATH = "Stress-Predict-Dataset/Raw_data"
SAVE_PATH = "save"

if not os.path.exists('Stress-Predict-Dataset'):
    # downloading dataset from given github link
    subprocess.run(
        "git clone https://github.com/italha-d/Stress-Predict-Dataset.git", shell=True)

    # removing readme file ( otherwise it will break the process of listing dir with os.listdir )
    subprocess.run("rm Stress-Predict-Dataset/Raw_data/Readme", shell=True)

    # create dir for saving data as csv files
    os.mkdir(SAVE_PATH)
else:
    print('You already cloned!')

# creating new dataframe obj for main four signals
acc = pd.DataFrame(columns=['id', 'X', 'Y', 'Z', 'datetime'])
bvp = pd.DataFrame(columns=['id', 'BVP', 'datetime'])
eda = pd.DataFrame(columns=['id', 'EDA', 'datetime'])
hr = pd.DataFrame(columns=['id', 'HR', 'datetime'])
temp = pd.DataFrame(columns=['id', 'TEMP', 'datetime'])

# fucntions for creating new dataframe with id and timestamp
def process_df(df, file):
    start_timestamp = df.iloc[0, 0]
    sample_rate = df.iloc[1, 0]
    new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
    new_df['id'] = file[-2:]
    new_df['datetime'] = [(start_timestamp + i / sample_rate) for i in range(len(new_df))]
    return new_df

names = {
    'ACC.csv': ['X', 'Y', 'Z'],
    'BVP.csv': ['BVP'],
    'EDA.csv': ['EDA'],
    'HR.csv': ['HR'],
    'TEMP.csv': ['TEMP'],
}
# main signals filenames that requesting from wearable sensors
desired_signals = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'TEMP.csv']

# looping though raw data folder and load data to previously created dataframes ( ex :- acc, eda, hr, temp)
for file in os.listdir(DATA_PATH):
    print(os.listdir(DATA_PATH))
    print(f'Processing {file}')
    for signal in os.listdir(os.path.join(DATA_PATH, file)):
        if signal in desired_signals:
            df = pd.read_csv(os.path.join(DATA_PATH, file, signal), names=names[signal], header=None)
            if not df.empty:
                if signal == 'ACC.csv':
                    acc = pd.concat([acc, process_df(df, file)])
                if signal == 'BVP.csv':
                    print(f'BVP Processing {file}')
                    bvp = pd.concat([bvp, process_df(df, file)])
                if signal == 'EDA.csv':
                    eda = pd.concat([eda, process_df(df, file)])
                if signal == 'HR.csv':
                    hr = pd.concat([hr, process_df(df, file)])
                if signal == 'TEMP.csv':
                    temp = pd.concat([temp, process_df(df, file)])
                    

# save combined data to csv files 
print('Saving Data ...')
acc.to_csv(os.path.join(SAVE_PATH, 'combined_acc.csv'), index=False)
bvp.to_csv(os.path.join(SAVE_PATH, 'combined_bvp.csv'), index=False)
eda.to_csv(os.path.join(SAVE_PATH, 'combined_eda.csv'), index=False)
hr.to_csv(os.path.join(SAVE_PATH, 'combined_hr.csv'), index=False)
temp.to_csv(os.path.join(SAVE_PATH, 'combined_temp.csv'), index=False)
print('Saving Completed!')

# find all ids from participants
ids = bvp['id'].unique()

# columns for after merging process 
columns = ['X', 'Y', 'Z', 'EDA','BVP', 'HR', 'TEMP', 'id', 'datetime']

# function merging data using pandas merge - https://pandas.pydata.org/docs/reference/api/pandas.merge.html
# will get null values after merging data using datetime as key
# padas DataFrame.fillna used for filling null values (ex : ffill, bfill) - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html 

def merge_parallel_without_null(id):
    print(f"Processing {id}")
    df = pd.DataFrame(columns=columns)

    acc_id = acc[acc['id'] == id]
    print('acc id', acc_id)
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    bvp_id = bvp[bvp['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)

    df = acc_id.merge(eda_id, on='datetime', how='outer')
    df = df.merge(bvp_id, on='datetime', how='outer')
    df = df.merge(temp_id, on='datetime', how='outer')
    df = df.merge(hr_id, on='datetime', how='outer')

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df

# function merging data using pandas merge - https://pandas.pydata.org/docs/reference/api/pandas.merge.html
# will get null values after merging data using datetime as key
# this fuction will not fill any null values so will have remove null values later

def merge_parallel_with_null(id):
    print(f"Processing {id}")
    df = pd.DataFrame(columns=columns)

    acc_id = acc[acc['id'] == id]
    print('acc id', acc_id)
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    bvp_id = bvp[bvp['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)

    df = acc_id.merge(eda_id, on='datetime', how='outer')
    df = df.merge(bvp_id, on='datetime', how='outer')
    df = df.merge(temp_id, on='datetime', how='outer')
    df = df.merge(hr_id, on='datetime', how='outer')

    return df

# variables for store data after merging process ( with null and without null )
# merging with bvp variable takes too much time.. process won't finish. ignore it for now
result_without_null = []
result_with_null = []

print("Merging data ...")
for i in ids:
    result_without_null.append(merge_parallel_without_null(i))
    result_with_null.append(merge_parallel_with_null(i))
print("Merging Completed!")

# creating dataframe with merging values
new_df_without_null = pd.concat(result_without_null, ignore_index=True)
new_df_with_null = pd.concat(result_with_null, ignore_index=True)

# save merged dataframe to csv for later usage
print("Saving merged data ...")
new_df_without_null.to_csv(os.path.join("save/merged_data_without_null.csv"), index=False)
new_df_with_null.to_csv(os.path.join("save/merged_data_with_null.csv"), index=False)
print("Saving merged data Completed!")

