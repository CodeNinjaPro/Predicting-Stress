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
from scipy import signal

# Load data from CSV file ( ex : participant no 02)
data = pd.read_csv('Stress-Predict-Dataset/Raw_data/S02/BVP.csv')

# Extract PPG signal from data
ppg_data = np.array(data[2:300])
ppg_data = ppg_data.reshape(298, )

print('shape ', ppg_data.shape)



# Step 1: Preprocessing
ppg_data = signal.detrend(ppg_data)  # remove linear trend
ppg_data = signal.medfilt(ppg_data, 11)  # apply median filter

# Step 2: Peak Enhancement
peaks, _ = signal.find_peaks(ppg_data, distance=32, height=0.3, prominence=0.1)

print('peaks', peaks)

# Step 3: Entropy-Based Signal Quality Index (ESQI)
def compute_esqi(signal):
    x_squared = signal ** 2
    esqi = -np.sum(x_squared * np.log(x_squared)) / len(signal)
    if np.isnan(esqi):
        esqi = 0
    return esqi

esqi = compute_esqi(ppg_data)

if esqi == 0:
    print('Signal quality is too low, unable to estimate respiratory rate')
else:
    # Step 4: Peak Detection
    rr_intervals = np.diff(peaks)
    mean_rr_interval = np.mean(rr_intervals)
    valid_peaks = [peaks[0]]  # always keep first peak
    for i in range(1, len(peaks)):
        if rr_intervals[i-1] >= 0.2 * mean_rr_interval:
            valid_peaks.append(peaks[i])

    # Step 5: Respiratory Rate Estimation
    ibi = np.diff(valid_peaks)
    freqs, psd = signal.welch(ibi, fs=1/mean_rr_interval, nperseg=64)
    resp_rate = freqs[np.argmax(psd)] * 60 * len(peaks) * 10
    print('Estimated respiratory rate:', resp_rate, 'breaths per minute')


# This function takes a Blood Volume Pulse (BVP) signal as input and calculates the respiratory rate in breaths per minute.
def get_respiratory_rate(bvp, sample_rate=64):

    try:
        # The BVP signal is first filtered using a bandpass filter with a cutoff frequency of 0.1-0.4 Hz.
        # Define bandpass filter parameters
        lowcut = 0.1  # Hz
        highcut = 0.4  # Hz
        order = 2

        # The filtered signal is then analyzed to detect peaks using a specified threshold and minimum distance between peaks.
        # Define peak detection parameters
        threshold = 0.3
        distance = 32  # Minimum distance between peaks (in samples)

        # Apply bandpass filter to BVP signal
        nyq = 0.3 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_bvp = filtfilt(b, a, bvp)

        # Find peaks in filtered signal
        peaks, _ = find_peaks(filtered_bvp, height=threshold, distance=distance)

        # Calculate respiratory rate (in breaths per minute)
        if len(peaks) > 0:
            respiratory_rate = 60 * len(peaks) / (len(filtered_bvp) / sample_rate) /11
            # print('Estimated respiratory rate:', respiratory_rate, 'breaths per minute')
        else:
            # If no peaks are detected, the function returns a value of -1.
            respiratory_rate = -1
            # print('No peaks detected in filtered signal')
    except:
        # If an error occurs during processing, the function also returns a value of -1.

        respiratory_rate = -1

    return respiratory_rate

# Load merged data from a CSV file into a Pandas DataFrame.
df = pd.read_csv('save/merged_data_without_null_with_lables.csv')
# Round the 'datetime' column values to the nearest integer.
df.datetime = df.datetime.round()
# Display the first few rows of the DataFrame to verify the data was loaded correctly.
df.head()

# Create an empty dataframe with only column names
new_df = pd.DataFrame(columns=['id','X','Y','Z','datetime','EDA','TEMP','HR','respr', 'y' ])

# Initialize variables
id = 0
datetime = 0
temp_bvp_arr = []
new_df = pd.DataFrame()

# Loop through each row in the dataframe
for index, row in df.iterrows():

    # Check if the id is the first one in the dataset
    if id == 0:
        # Store values from the first row
        id = row['id']
        X = row['X']
        Y = row['Y']
        Z = row['Z']
        datetime = row['datetime']
        EDA = row['EDA']
        BVP = row['BVP']
        TEMP = row['TEMP']
        HR = row['HR']
        y = row['y']
    else:
        # Check if the current row has the same id and datetime as the previous row
        if id == row['id'] and datetime == row['datetime']:
            # Add BVP value to temporary array
            temp_bvp_arr.append(row['BVP'])
            X = row['X']
            Y = row['Y']
            Z = row['Z']
            datetime = row['datetime']
            EDA = row['EDA']
            TEMP = row['TEMP']
            HR = row['HR']
            y = row['y']
        else:
            # Calculate respiratory rate from BVP values in temporary array
            respiratory_rate = get_respiratory_rate(np.array(temp_bvp_arr))

            # Check if a valid respiratory rate was calculated
            if respiratory_rate != -1:
                # Create a new row for the output dataframe with respiratory rate and other values
                new_row = {'id': id, 'X': X, 'Y': Y, 'Z': Z, 'datetime': datetime,
                           'EDA': EDA, 'TEMP': TEMP, 'HR': HR, 'respr': respiratory_rate, 'y': y}
                # Add the new row to the output dataframe
                new_df = pd.concat([new_df, pd.DataFrame(
                    new_row, index=[0])], ignore_index=True)

            # Clear the temporary BVP array and store values from the current row
            temp_bvp_arr.clear()
            id = row['id']
            temp_bvp_arr.append(row['BVP'])
            X = row['X']
            Y = row['Y']
            Z = row['Z']
            datetime = row['datetime']
            EDA = row['EDA']
            TEMP = row['TEMP']
            HR = row['HR']
            y = row['y']

# Write the final output dataframe to a CSV file
new_df.to_csv('save/finalise_dataset_with_resper_last.csv', index=False)

# Print a message indicating that the script has finished running
print('Resper calculation completed')
