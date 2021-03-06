import symbionic
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import Features
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def get_data(directory_name, window_length, step_time):
    emg_data = symbionic.EmgData()
    emg_data.load(directory_name + 'raw1.bin', gesture='g1')
    emg_data.load(directory_name + 'raw2.bin', gesture='g2')
    emg_data.load(directory_name + 'raw3.bin', gesture='g3')
    emg_data.load(directory_name + 'raw4.bin', gesture='g4')
    emg_data.load(directory_name + 'raw5.bin', gesture='g5')
    emg_data.load(directory_name + 'raw6.bin', gesture='g6')
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


def get_data_multiple_folders(directory_name, window_length, step_time):
    total_data = []
    total_labels = []
    total_dt = []
    for root, dirs, files in os.walk(directory_name):
        for dir in dirs[:3]:
            emg_data = symbionic.EmgData()
            print(dir)
            seq = os.listdir(directory_name + '\\' + dir)
            i = 1
            for file in seq:
                emg_data.load(directory_name + '\\' + dir + '\\' + file, gesture='g' + str(i))
                # print(directory_name + '\\' + dir + '\\' + file + 'gesture=g'+str(i))
                i = i + 1
            emg_data.label_patterns()
            result = emg_data.get_training_samples(window=window_length, step=step_time)
            data = result['X']
            labels = result['y']
            dt = result['dt']
            selected_indices = np.where((dt > -0.2) & (dt < 0.90))
            data = data[selected_indices]
            labels = labels[selected_indices]
            dt = dt[selected_indices]

            total_data.append(data)
            total_labels.append(labels)
            total_dt.append(dt)
            del emg_data

    flat_data_list = [item for sublist in total_data for item in sublist]
    flat_labels_list = [item for sublist in total_labels for item in sublist]
    flat_dt_list = [item for sublist in total_dt for item in sublist]
    numpy_data = np.asarray(flat_data_list)
    numpy_labels = np.asarray(flat_labels_list)
    numpy_dt = np.asarray(flat_dt_list)
    return numpy_data, numpy_labels, numpy_dt


def get_features(window_length, input_data):
    Fs = 650
    T = 1 / Fs
    t = window_length
    N = Fs * t
    start_time = time.time()
    CS35 = Features.CS35(input_data, N, Fs)
    end_time = time.time() - start_time
    print("preprocessing time of 1 window in ms:" + str(end_time / data.shape[0] * 1000))
    return CS35


def train_algorithm(features_train, labels_train, dt):
    X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(features_train, labels_train, dt, test_size=0.2, random_state=9)
    model = ExtraTreesClassifier(bootstrap=False, criterion='gini', max_depth=40, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=600)
    Trees = model.fit(X_train, y_train)
    return Trees, X_test, y_test#, dt_train, dt_test


def predict_algorithm(trained_algorithm, features, labels):
    predictions = trained_algorithm.predict(features)
    pred_labels = np.argmax(predictions, axis=0)
    forest_acc = accuracy_score(labels, predictions) * 100
    return forest_acc
    #print("Fit accuracy is {:.1f}%".format(forest_acc))


folder_train = r'/home/munns/Desktop/Symbionic project/Symbionic-algorithm-development/sample data/new/train/train1/'
#folder_train_more_data = r'C:\Users\Paul-PC\OneDrive - Avans Hogeschool\TMC\Symbionic ai-development\sample data\new\train'
folder_test = r'/home/munns/Desktop/Symbionic project/Symbionic-algorithm-development/sample data/new/test/train2/'

# Note: the input window size is in seconds
window_time = 0.45
step = 0.10

#training
data, labels, dt = get_data(folder_train, window_time, step)
CS35_feature = get_features(window_time, data)
trained_algorithm, features_test, labels_test = train_algorithm(CS35_feature, labels, dt)
accu_train = predict_algorithm(trained_algorithm, features_test, labels_test)

#testing
data, labels, dt = get_data(folder_test, window_time, step)
CS35_feature = get_features(window_time, data)
accu_test = predict_algorithm(trained_algorithm, CS35_feature, labels)

print("window_size = " + str(window_time) + " and step_size = " + str(step) + " accuracy(train,test) = " + str(
    accu_train) + ' ' + str(accu_test))

#result_df = pd.DataFrame(result_lst, columns=['window-size', 'step-size', 'train_accu', 'test-accu'])
#result_df.to_csv('window_step_size_train1_v2.csv')
