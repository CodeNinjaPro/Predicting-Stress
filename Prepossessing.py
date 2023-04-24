# importing libraries 
import os
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
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler


# Load the dataset into a Pandas DataFrame
all_df = pd.read_csv('save/finalise_dataset_with_resper_last.csv')
# Ensuring that leave a subset of data separate from the exploration to avoid overfitting.
# Randomly sample 70% of your dataframe
df = all_df.sample(frac=0.7)

# Here is rest 30% data 
df_test = all_df.loc[~all_df.index.isin(df.index)]
# Save it as csv for later
df_test.to_csv('save/test_dataset.csv', index=False)

# Check the balance of the target variable
class_counts = df['y'].value_counts()
print(class_counts)

# Separate the features and the target variable
X = df.drop('y', axis=1)
y = df['y']

# Check the balance of the target variable
class_counts = y.value_counts()
print('Class distribution before balancing:')
print(class_counts)

# Undersampling
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)
df_undersampled = pd.concat([X_resampled, y_resampled], axis=1)

# Oversampling
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)
df_oversampled = pd.concat([X_resampled, y_resampled], axis=1)

# Hybrid Methods
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
df_smote_tomek = pd.concat([X_resampled, y_resampled], axis=1)

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
df_smote_enn = pd.concat([X_resampled, y_resampled], axis=1)

# Class Weights
class_weights = compute_class_weight('balanced', classes=y.unique(), y=y)
class_weights_dict = dict(zip(y.unique(), class_weights))
df_weighted = df.copy()
df_weighted['class_weight'] = df_weighted['y'].apply(lambda x: class_weights_dict[x])

# Save the balanced datasets to CSV files
df_undersampled.to_csv('save/undersampled_dataset.csv', index=False)
df_oversampled.to_csv('save/oversampled_dataset.csv', index=False)
df_smote_tomek.to_csv('save/smote_tomek_dataset.csv', index=False)
df_smote_enn.to_csv('save/smote_enn_dataset.csv', index=False)
df_weighted.to_csv('save/weighted_dataset.csv', index=False)

# Check the balance of the target variable after balancing
print('Class distribution after balancing using RandomUnderSampler:')
print(df_undersampled['y'].value_counts())

print('Class distribution after balancing using SMOTE:')
print(df_oversampled['y'].value_counts())

print('Class distribution after balancing using SMOTE + Tomek links:')
print(df_smote_tomek['y'].value_counts())

print('Class distribution after balancing using SMOTE + ENN:')
print(df_smote_enn['y'].value_counts())

print('Class distribution after balancing using class weights:')
print(df_weighted['y'].value_counts())

def visualise_outlier(df, col, visualise=True):
    """
    Visualize potential outliers using boxplots and return a boolean value indicating if outliers were found.

    Parameters:
        df (DataFrame): the dataframe to check for outliers
        col (str): the name of the column to check for outliers

    Returns:
        bool: True if outliers were found, False otherwise
    """
    if visualise:

      # Define the box and outlier properties
      boxprops = dict(linestyle='-', linewidth=2, color='black')
      flierprops = dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red')

      # Create subplots for each variable
      fig, axs = plt.subplots(ncols=1, figsize=(20,10))

      # Plot the boxplot and set the title
      sns.boxplot(data=df, x=col, boxprops=boxprops, flierprops=flierprops)
      axs.set_title(col)

      # Adjust the layout and show the plot
      plt.tight_layout()
      plt.show()

    # Check for outliers and return the result
    q1, q3 = np.percentile(df[col], [25, 75])
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]

    if outliers.empty:
        return False
    else:
        return True


col = 'X'

status = visualise_outlier(df_smote_enn, col)
if status:
  print('Outliers found in ', col)
else:
  print('Outliers not found in ', col, ' good to go')

# Winsorize the data
# Replace the outliers in a pandas DataFrame with the nearest values that are within the 5th and 95th percentiles

# Make a copy of the DataFrame to preserve the original data
win_df_smote_enn = df_smote_enn.copy()

# Define the columns that you want to winsorize
""" the id and datetime columns are typically not subject to outlier removal, 
    because they represent unique identifiers and timestamps, respectively, 
    and not numerical data that can contain outliers. """

# We skip winsorizing the id and datetime columns as they are unique identifiers and timestamps, respectively
columns_to_winsorize = ['X',	'Y',	'Z',	'EDA',	'TEMP',	'HR',	'respr']

# Winsorize the specified columns in the DataFrame
win_values = winsorize(win_df_smote_enn[columns_to_winsorize].values, limits=[0.25, 0.25])

# Replace the original columns with the winsorized values
win_df_smote_enn[columns_to_winsorize] = win_values

# Compare the original and winsorized values
for column in columns_to_winsorize:
    outliers_removed = np.sum(df_smote_enn[column] != win_df_smote_enn[column])
    print(f'Outliers removed from {column}: {outliers_removed}')



# *** Standardization and normalization ***

# Make a copy of the DataFrame to preserve the original data
data = win_df_smote_enn.copy()

# select the numeric variables to be normalized
num_vars = ['X', 'Y', 'Z', 'EDA', 'TEMP', 'HR', 'respr']

# create a MinMaxScaler object and fit it to the data
scaler = MinMaxScaler()
scaler.fit(data[num_vars])

# transform the selected variables using the fitted scaler
data[num_vars] = scaler.transform(data[num_vars])

# Saving preprocessed  data into csv
data.to_csv('save/preprocessed_data.csv', index=False)