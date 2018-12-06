import symbionic
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import FeaturesComparison
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Note: the input window size is in seconds
window_time = 0.45
#step = 0.11


def get_data(directory_name, window_length, step_time):
    emg_data = symbionic.EmgData()
    emg_data.load(directory_name + 'raw1.bin', gesture='g1')
    emg_data.load(directory_name + 'raw2.bin', gesture='g2')
    emg_data.load(directory_name + 'raw3.bin', gesture='g3')
    emg_data.load(directory_name + 'raw4.bin', gesture='g4')
    emg_data.load(directory_name + 'raw5.bin', gesture='g5')
    emg_data.label_patterns()
    result = emg_data.get_training_samples(window=window_length, step=step_time)
    data = result['X']
    labels = result['y']
    dt = result['dt']
    selected_indices = np.where((dt > -0.2) & (dt < 0.90))
    data = data[selected_indices]
    labels = labels[selected_indices]
    dt = dt[selected_indices]
    return data, labels, dt


folder_train = r'C:\Users\Paul-PC\OneDrive - Avans Hogeschool\TMC\Schooldocuments\Bezig\Productreport\sample data report Paul\new\train1\\'


def get_features(window_length, data, labels):
    Fs = 650
    T = 1 / Fs
    t = window_length
    N = Fs * t
    start_time = time.time()
    CS35 = FeaturesComparison.CS35(data, N, Fs)
    end_time = time.time() - start_time
    print("preprocessing time of 1 window in ms:" + str(end_time / data.shape[0] * 1000))
    return CS35


def train_algorithm(features_train, labels_train, dt):
    X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(features_train, labels_train, dt, test_size=0.2, random_state=9)
    model = ExtraTreesClassifier(n_estimators=400, max_features="log2", criterion="entropy")
    Trees = model.fit(X_train, y_train)
    return Trees


def


start_time = time.time()
forest_predictions = Trees.predict(X_test)
end_time = time.time() - start_time
print("prediction time of 1 window in ms:" + str(end_time/X_test.shape[0]*1000))
forest_pred_labels = np.argmax(forest_predictions,axis=0)
forest_acc = accuracy_score(y_test_RF,forest_predictions)*100

print("Fit accuracy is {:.1f}%".format(forest_acc))