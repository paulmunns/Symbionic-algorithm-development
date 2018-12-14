import symbionic
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import Features
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd



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


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    average_error = np.mean(errors)
    accuracy = accuracy_score(y_test, predictions) * 100
    return average_error, accuracy

# numbers of trees
n_estimators = [int(x) for x in np.linspace(start=200, stop = 2000, num=10)]
# numbers of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
criterion = ['gini', 'entropy']
# maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}

folder_train = r'/home/munns/Desktop/Symbionic project/Symbionic-algorithm-development/sample data/new/train/train1/'
window_time = 0.45
step = 0.10
data, labels, dt = get_data(folder_train, window_time, step)
CS35_feature = get_features(window_time, data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(CS35_feature, labels, dt, test_size=0.2, random_state=9)

model = ExtraTreesClassifier()
model_random = GridSearchCV(estimator = model, param_grid = random_grid, cv = 4, verbose=8, n_jobs = -1, return_train_score=True)
Trees = model_random.fit(X_train, y_train)
best_params = Trees.best_params_
best_grid = Trees.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)

with open('best_params_ExtraTrees', 'wb') as f:
    pickle.dump(str(best_params), f)

with open('best_params_accuracy', 'wb') as f:
    pickle.dump(str(grid_accuracy), f)

resulting = Trees.cv_results_
df = pd.DataFrame(resulting)
df.to_csv('CV-scorings.csv', encoding='utf-8')
