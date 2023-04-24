# Predicting Stress Levels Using Wearable Device Data

This study investigates whether it is possible to identify human stress with the currently available sensors on the Empatica E4 watch. This study used a few deep learning models, which are Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Autoencoder, and Generative Adversarial Networks (GAN) evaluated alongside several machine learning models, including Forest and a few more machine learning models. Eight different signals comprised the dataset used in this study, which Empatica E4 produced watch.
The Decision Tree model came in second with an accuracy of 95.33
These findings imply that machine learning methods, particularly Random Forest and Decision Tree are superior to deep learning models at identifying human stress from Empatica E4 watch sensor data. The study’s shortcomings, with Fine tuned Random Forest model, can be used to detect stress with very low False Negative prediction. Also, additional investigation into the application of various deep learning architectures and the effect of feature selection on the effectiveness of machine learning models is advised.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contact](#Contact)

## Getting Started

I strongly suggest to run notebook file in colab. notebook files included all content

### Installation

Clone the repository: git clone https://github.com/CodeNinjaPro/Predicting-Stress.git

Install the required packages: pip install -r requirements.txt



## Usage

Run the script : python LoadingData.py

Run the script : python Labelling.py

Run the script : python CalculateRespiratory.py

Run the script : python Prepossessing.py

Run the script : python ML.py

Run the script : python DL.py

## Contact

name : Ahangama Withanage Roshan Darshana Madhushanka

email : ra22242@essex.ac.uk

project : Predicting Stress Levels Using Wearable Device Data: A Data Analysis Report

*** Important - use google colab to run  all_step_notebook.ipynb ***






