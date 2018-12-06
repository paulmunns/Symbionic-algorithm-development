"""
Symbionic project
Written by: Paul Munns
Usage: Implement different feature extraction methods on a 3D-array of EMG signals
"""

import numpy as np
import symbionic

#integrated absolute value
def IAV(data):
    data_abs = np.abs(data)
    valuea = np.apply_along_axis(lambda x: np.trapz(x), 1, data_abs)
    return valuea

#Root mean square
def RMS(data):
    valueb = np.apply_along_axis(lambda x: np.sqrt(np.mean(x**2)), 1, data)
    return valueb

#Mean absolute value
def MAV(data):
    data_abs = np.abs(data)
    valuec = np.apply_along_axis(lambda x: np.mean(x), 1, data_abs)
    return valuec

#Hilbert transform (upper envelope)
def envelope_methode(data):
    envelopes = np.apply_along_axis(lambda x: symbionic.calc_envelope_no_filter(x,smooth=51),1,data)
    envelopes = np.swapaxes(envelopes,1,2)
    flattened_envelopes = envelopes.reshape((data.shape[0],data.shape[1]*data.shape[2]))
    return flattened_envelopes

#zero crossing count the amount of times the zero is crossed
#its an estimation of the median frequency
def ZC(data):
    emg_ar = []
    for m in range(data.shape[0]):
        for n in range(data.shape[2]):
            value = ((data[m, :-1, n] * data[m, 1:, n]) < 0).sum()
            emg_ar = np.append(emg_ar, value)
    emg_ar = np.reshape(emg_ar, (-1, 8))
    return emg_ar

#Frequency domain transformation
def FFT_domain(data, N, Fs):
    fft_data = np.apply_along_axis(lambda x: np.fft.fft(x)[0:int(N/2)]/N, 1, data)
    fft_data = 2 * fft_data
    Pxx = np.abs(fft_data)
    f = Fs * np.arange((N / 2)) / N
    return Pxx, f

#Mean value of Frequency spectrum
def FFT_mean(data, N, Fs):
    x, y = FFT_domain(data, N, Fs)
    mean_value_fft = np.apply_along_axis(lambda x: np.mean(x), 1, x)
    return mean_value_fft

#Highest frequency amplitude
def FFT_max(data, N, Fs):
    x, y = FFT_domain(data, N, Fs)
    max_value_fft = np.apply_along_axis(lambda x: np.max(x), 1, x)
    return max_value_fft

#return absolute value
def absmax_at_end(data):
    absmax = np.apply_along_axis(lambda x: np.max(abs(x[-50:])), 1, data)
    return absmax

#Combination of MAV, WL, ZC, SSC, RMS, AR6
def CS3(data, N, Fs):
    IAV_method = IAV(data)
    RMS_method = RMS(data)
    MAV_method = MAV(data)
    FFT_mean_method = FFT_mean(data,N, Fs)
    FFT_max_method = FFT_max(data, N, Fs)
    Absmax_method = absmax_at_end(data)
    ZC_method = ZC(data)
    CS3_value = np.append(IAV_method, RMS_method, axis=1)
    CS3_value = np.append(CS3_value, MAV_method, axis=1)
    CS3_value = np.append(CS3_value, FFT_mean_method, axis=1)
    CS3_value = np.append(CS3_value, FFT_max_method, axis=1)
    CS3_value = np.append(CS3_value, Absmax_method, axis=1)
    CS3_value = np.append(CS3_value, ZC_method, axis=1)
    return CS3_value

#IAV, RMS, MAV
def CS32(data):
    IAV_method = IAV(data)
    RMS_method = RMS(data)
    MAV_method = MAV(data)
    CS32_value = np.append(IAV_method, RMS_method, axis=1)
    CS32_value = np.append(CS32_value, MAV_method, axis=1)
    return CS32_value

#IAV, RMS, MAV, FFT_mean, FFT_max
def CS33(data, N, Fs):
    IAV_method = IAV(data)
    RMS_method = RMS(data)
    MAV_method = MAV(data)
    FFT_mean_method = FFT_mean(data,N, Fs)
    FFT_max_method = FFT_max(data, N, Fs)
    CS33_value = np.append(IAV_method, RMS_method, axis=1)
    CS33_value = np.append(CS33_value, MAV_method, axis=1)
    CS33_value = np.append(CS33_value, FFT_mean_method, axis=1)
    CS33_value = np.append(CS33_value, FFT_max_method, axis=1)
    return CS33_value

#IAV, RMS, MAV, FFT_mean, FFT_max, Absmax
def CS34(data, N, Fs):
    IAV_method = IAV(data)
    RMS_method = RMS(data)
    MAV_method = MAV(data)
    FFT_mean_method = FFT_mean(data,N, Fs)
    FFT_max_method = FFT_max(data, N, Fs)
    Absmax_method = absmax_at_end(data)
    CS34_value = np.append(IAV_method, RMS_method, axis=1)
    CS34_value = np.append(CS34_value, MAV_method, axis=1)
    CS34_value = np.append(CS34_value, FFT_mean_method, axis=1)
    CS34_value = np.append(CS34_value, FFT_max_method, axis=1)
    CS34_value = np.append(CS34_value, Absmax_method, axis=1)
    return CS34_value

#Combination of WL, RMS, SampEn, CC4
def CS35(data, N, Fs):
    IAV_method = IAV(data)
    RMS_method = RMS(data)
    MAV_method = MAV(data)
    FFT_mean_method = FFT_mean(data,N, Fs)
    FFT_max_method = FFT_max(data, N, Fs)
    Absmax_method = absmax_at_end(data)
    ZC_method = ZC(data)
    hilbert_method = envelope_methode(data)
    CS35_value = np.append(IAV_method, RMS_method, axis=1)
    CS35_value = np.append(CS35_value, MAV_method, axis=1)
    CS35_value = np.append(CS35_value, FFT_mean_method, axis=1)
    CS35_value = np.append(CS35_value, FFT_max_method, axis=1)
    CS35_value = np.append(CS35_value, Absmax_method, axis=1)
    CS35_value = np.append(CS35_value, ZC_method, axis=1)
    CS35_value = np.append(CS35_value, hilbert_method, axis=1)
    return CS35_value