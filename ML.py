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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle

# read data from previous steps
data = pd.read_csv('save/preprocessed_data.csv')
# create x and y values for machine learning methods

features_x = ['X', 'Y', 'Z', 'datetime', 'EDA', 'TEMP', 'HR', 'respr']
features_y = ['y']

X = data[features_x]
y = data[features_y]

# convert y to a 1D array using ravel()
y_1d = y.to_numpy().ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_1d, test_size=0.2, random_state=42)

# Define the models to test
models = [
    {'name': 'Logistic Regression', 'model': LogisticRegression()},
    {'name': 'Decision Tree', 'model': DecisionTreeClassifier()},
    {'name': 'K-Nearest Neighbors', 'model': KNeighborsClassifier()},
    {'name': 'Gaussian Naive Bayes', 'model': GaussianNB()},
    {'name': 'Support Vector Machine', 'model': SVC()},
    {'name': 'Random Forest', 'model': RandomForestClassifier()},
    {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier()}
]

# Train and test each model
train_times = []
accuracies = []
for model in tqdm(models):
    start_time = time.time()
    clf = model['model']
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    train_times.append(train_time)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{model['name']}: accuracy = {accuracy:.3f}, train time = {train_time:.3f} seconds")

# Display a table of the results
results_df = pd.DataFrame({'Model': [model['name'] for model in models],
                           'Train Time (s)': train_times,
                           'Accuracy': accuracies})
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print('\nResults:')
print(results_df)


# Plot the results
fig, ax = plt.subplots()
bars = ax.bar(np.arange(len(models)), accuracies)
ax.set_xticks(np.arange(len(models)))
# Comment below line just because names are too long and it will looks like messy 
# ax.set_xticklabels([model['name'] for model in models])
ax.set_ylim(0, 1)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Models')
for i, acc in enumerate(accuracies):
    ax.text(i, acc+0.02, f"{acc:.4f}", ha='center')
# Add model name inside each bar
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width() / 2., 0.3, models[i]['name'], ha='left', va='center', rotation=90)

plt.show()


# According to previous result Random Forest is the best ml model for this dataset
# Now just trying to increase accuracy a little 
# initialize a random forest classifier with 200 trees
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)

# fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# use the trained classifier to predict the test data
y_pred = rf_classifier.predict(X_test)

# evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1-Score:", f1)


conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)


# perform 10-fold cross-validation and calculate the mean accuracy and F1 score
cv_accuracy = cross_val_score(rf_classifier, X_train, y_train, cv=10, scoring='accuracy').mean()
cv_f1 = cross_val_score(rf_classifier, X_train, y_train, cv=10, scoring='f1').mean()

print("Cross-validated Accuracy:", cv_accuracy)
print("Cross-validated F1-Score:", cv_f1)


# The company is worried about false negatives and wants to show that the device won't miss stressful periods. 
# predict probabilities for the test data
y_prob = rf_classifier.predict_proba(X_test)

# adjust threshold to increase sensitivity to positive class
y_pred_new = (y_prob[:, 1] > 0.3).astype(int)

# evaluate the accuracy of the classifier with new predictions
accuracy_new = accuracy_score(y_test, y_pred_new)
f1_new = f1_score(y_test, y_pred_new)
print("New Accuracy:", accuracy_new)
print("New F1-Score:", f1_new)

# print confusion matrix for new predictions
conf_mat_new = confusion_matrix(y_test, y_pred_new)
print(conf_mat_new)

# save the trained classifier to a file
with open('save/rf_classifier.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)