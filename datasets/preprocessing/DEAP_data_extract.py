#----------------------------------------------Description-----------------------------------------------------------
# In this part, we preprocess the original EEG data of DEAP dataset.
# we first use band pass filter, and then calculate the DE features of the base data and the trial data.
# Secondly, we get the labels of each segment.
# Finally, we define the WGN generation and the data normalization function, but it seems that we did not use them.
# We shoule check the shape of the output data later.
#-------------------------------------------------E N D--------------------------------------------------------------

import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter

# Design a band pass filter.
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Use the band pass filter to eliminate the noise, the band is 4Hz ~ 45Hz.
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Read data from the original data of DEAP.
def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data

# Input is signal, compute its differential entropy.
# when compute DE, we use standard variance, why here is variance?
def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    de = math.log(2 * math.pi * math.e * variance) / 2
    return de

def decompose(file):
    # trial x channel x sample
    start_index = 384  # 3s pre-trial signals
    # read data from .mat file
    data = read_file(file)
    shape = data.shape
    frequency = 128

    decomposed_de = np.empty([0, 4, 120])

    base_DE = np.empty([0, 128])
    
    # for each trial of one subject
    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])

        temp_de = np.empty([0, 120])

        for channel in range(32):
            trial_signal = data[trial, channel, 384:]
            # base_signal is the first 3 seconds of the data
            base_signal = data[trial, channel, :384]
            # ****************compute base DE****************
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)
            
            # We first divide the baseline signals into 6 segments with 0.5 s
            # Extract DE features from four frequency bands of each segment
            # six segments --> sum and average
            base_theta_DE = (compute_DE(base_theta[:64]) + compute_DE(base_theta[64:128]) + compute_DE(
                base_theta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_alpha_DE = (compute_DE(base_alpha[:64]) + compute_DE(base_alpha[64:128]) + compute_DE(
                base_alpha[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_beta_DE = (compute_DE(base_beta[:64]) + compute_DE(base_beta[64:128]) + compute_DE(
                base_beta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_gamma_DE = (compute_DE(base_gamma[:64]) + compute_DE(base_gamma[64:128]) + compute_DE(
                base_gamma[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)
            
            # ****************compute trial DE****************
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)
            
            # 0.5s each segment, 60s each trial, hence we get 120 windows of each trial(channel)
            for index in range(120):
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 64:(index + 1) * 64]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 64:(index + 1) * 64]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 64:(index + 1) * 64]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 64:(index + 1) * 64]))
                
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])
        
        # print(temp_de.shape)
        temp_trial_de = temp_de.reshape(-1, 4, 120)
        # trial DE of each trial
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)
        # base DE of each trial
        base_DE = np.vstack([base_DE, temp_base_DE])
    # all trials    
    decomposed_de = decomposed_de.reshape(-1, 32, 4, 120).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)
    print("base_DE shape:", base_DE.shape)
    print("trial_DE shape:", decomposed_de.shape)
    return base_DE, decomposed_de


def get_labels(file):
    # 0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  # valence labels
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  # arousal labels
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    for i in range(len(valence_labels)):
        for j in range(0, 120):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("labels:", final_arousal_labels.shape)
    return final_arousal_labels, final_valence_labels

# Generates white Gaussian noise (WGN), snr is signal-to-noise ratio.
def wgn(x, snr):
    # Converts the SNR from dB to a linear scale
    snr = 10 ** (snr / 10.0)
    # Calculates the power of the original signal "x"
    xpower = np.sum(x ** 2) / len(x)
    # Calculates the noise power required to achieve the desired snr by dividing the signal power by the snr.
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def feature_normalize(data):
    # find the nonzero elements, and calculate their mean and std 
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized


if __name__ == '__main__':
    dataset_dir = "./datasets/DEAP/"

    result_dir = "./datasets/DEAP_all_0p5/"
    if os.path.isdir(result_dir) == False:
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        base_DE, trial_DE = decompose(file_path)
        arousal_labels, valence_labels = get_labels(file_path)
        sio.savemat(result_dir + "DE_" + file,
                    {"base_data": base_DE, "data": trial_DE, "valence_labels": valence_labels,
                     "arousal_labels": arousal_labels})
