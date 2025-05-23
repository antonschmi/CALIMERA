import numpy as np
import pandas as pd
import joblib

from calimera import CALIMERA
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score


def load_example_data():
    # example data from https://timeseriesclassification.com/
    X_train, y_train = load_from_tsfile_to_dataframe(
        '/media/storage/users/go98kef/Cricket/Cricket_TRAIN.ts'
    )
    X_train = np.asarray([[[v for v in channel] for channel in sample] for sample in X_train.to_numpy()])
    X_test, y_test = load_from_tsfile_to_dataframe(
        '/media/storage/users/go98kef/Cricket/Cricket_TEST.ts'
    )
    X_test = np.asarray([[[v for v in channel] for channel in sample] for sample in X_test.to_numpy()])
    return X_train, y_train, X_test, y_test




def load_bugsense_data():
    data = pd.read_csv('/media/storage/users/go98kef/TimeSeriesConversion/time_series_data_all_classes_norm.csv', header=None)
    labels = pd.read_csv('/media/storage/users/go98kef/TimeSeriesConversion/time_series_labels_all_classes_norm.csv', usecols=[0], header=None)
    labels_grouped = labels[0].astype(str).to_numpy()[::80]

    # Calculate number of complete groups of 80
    n_samples = len(data) // 80

    # Reshape data into groups of 80
    data_grouped = data.values.reshape(n_samples, 80, 24).transpose(0, 2, 1)
    

    # Generate random split indices (70% train, 30% test)
    train_size = int(0.7 * n_samples)
    test = int(0.3 * n_samples/2)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+test]
    test_indices = indices[train_size+test:]
    

    # Split data and labels
    X_train = data_grouped[train_indices]
    X_test = data_grouped[test_indices]
    y_train = labels_grouped[train_indices]
    y_test = labels_grouped[test_indices]
    X_val = data_grouped[val_indices]
    y_val = labels_grouped[val_indices]


    return X_train, y_train.squeeze(), X_val, y_val.squeeze(), X_test, y_test.squeeze()

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_bugsense_data()
    accuracy_scores = []
    earliness_scores = []
    cost_scores = []
    
    for delay_penalty in range(1,11):
        model = CALIMERA(delay_penalty=delay_penalty/10)
        model.fit(X_train, y_train)

        stop_timestamps, y_pred = model.test(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
        cost = 1.0 - accuracy + delay_penalty/10 * earliness
        print(f'Accuracy: {accuracy}\nEarliness: {earliness}\nCost: {cost}')

        accuracy_scores.append(accuracy)
        earliness_scores.append(earliness)
        cost_scores.append(cost)
        model_filename = f'saved_models/calimera_model_delay_{delay_penalty / 10}.joblib'
        joblib.dump(model, model_filename)
    

    print(f'Accuracy scores: {accuracy_scores}')
    print(f'Earliness scores: {earliness_scores}')
    print(f'Cost scores: {cost_scores}')