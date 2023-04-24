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

new_df_without_null = pd.read_csv('save/merged_data_without_null.csv')
new_df_with_null = pd.read_csv('save/merged_data_with_null.csv')

# dictionary for saving timestamp particular participants
data = {}

# looping though folders for record timestamp for particular participants
for file in os.listdir(DATA_PATH):
    for signal in os.listdir(os.path.join(DATA_PATH, file)):
        if 'tags_' in signal:
            test_df = pd.read_csv(os.path.join(
                DATA_PATH, file, signal), header=None)

            temp_data = []
            for index, row in test_df.iterrows():
                temp_data.append(int(row))
            data[int(signal[7:8])] = temp_data

# find labes for merged dataframe from given tag files ( ex: tags_02.csv )
print('This process will take time. Please wait! here listen some music ',
      'https://www.youtube.com/watch?v=VCamMhh9pMA')
start_time = time.time()

# variables for holding lables
y_without_null = []

# looping dataframe row by row ( dataframe without null values)
for index, row in new_df_without_null.iterrows():
    # just coping data dictionary ( not referenceing it )
    temp_list = data[row['id']].copy()

    # here logic is simple
    # first add timestamp to tags list
    temp_list.append(row['datetime'])
    # then we sort it
    temp_list.sort()

    # if added timestamp's index is 0, participant is not in stress
    if temp_list.index(row['datetime']) == 0:
        y_without_null.append(0)
    # if added timestamp's index is more than 6, it means participant do task again. so he face same question and task that
    # he did before, I assume he will no stress by answering same question again. because he already knew the answer
    elif temp_list.index(row['datetime']) > 5:
        y_without_null.append(0)
    # if added timestamp's index is even number, witch meas participant press button twice. so participant no in stressed
    elif temp_list.index(row['datetime']) % 2 == 0:
        y_without_null.append(0)
    # if added timestamp's index is even number, participant is not in stress
    else:
        y_without_null.append(1)
    temp_list = []

end_time = time.time()

print("Time taken for labelling (in seconds) : ", end_time - start_time)

new_df_without_null['y'] = y_without_null
new_df_without_null.to_csv(os.path.join(
    "save/merged_data_without_null_with_lables.csv"), index=False)

saving_time = time.time()

print("Time taken for saving data to csv (in seconds) : ", saving_time - end_time)


# find labes for merged dataframe from given tag files ( ex: tags_02.csv )
print("This won't will take much time! anyway here listen some music ",
      'https://songsara.net/58262/')
start_time = time.time()

# variables for holding lables
y_with_null = []

# removing null raws
new_df_with_null = new_df_with_null.dropna(how='any', axis=0)
print(new_df_with_null.info())

# looping dataframe row by row ( dataframe without null values)
for index, row in new_df_with_null.iterrows():
    # just coping data dictionary ( not referenceing it )
    temp_list = data[row['id']].copy()

    # here logic is simple
    # first add timestamp to tags list
    temp_list.append(row['datetime'])
    # then we sort it
    temp_list.sort()
    # if added timestamp's index is 0, participant is not in stress
    if temp_list.index(row['datetime']) == 0:
        y_with_null.append(0)
    # if added timestamp's index is more than 6, it means participant do task again. so he face same question and task that
    # he did before, I assume he will no stress by answering same question again. because he already knew the answer
    # if added timestamp's index is even number, witch meas participant press button twice. so participant no in stressed ( he is in interval time)
    elif temp_list.index(row['datetime']) % 2 == 0 or temp_list.index(row['datetime']) > 5:
        y_with_null.append(0)
    # if added timestamp's index is even number, participant is not in stress
    else:
        y_with_null.append(1)
    temp_list = []

end_time = time.time()

print("Time taken for labelling (in seconds) : ", end_time - start_time)

new_df_with_null['y'] = y_with_null
new_df_with_null.to_csv(os.path.join(
    "save/merged_data_with_null_with_lables.csv"), index=False)

saving_time = time.time()

print("Time taken for saving data to csv (in seconds) : ", saving_time - end_time)
