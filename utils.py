import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import numpy as np


def load_dataset(full_path, num_label_sets):
    # load the dataset as a numpy array
    data = pd.read_csv(full_path)

    if data.isnull().values.any():
        raise ValueError('Empty or nan cells exist in the dataset')

    # if there are any categorical features, convert them to numerical features
    for column in data.columns:
        if data[column].dtype == 'object':
            data = categorical_to_numerical(data, column)

    # split into input and output elements
    X = data.iloc[:, :-num_label_sets]
    X = X.to_numpy()

    # prepare label sets
    Y = list()
    for i in range(num_label_sets):
        labels = data.iloc[:, -num_label_sets + i]
        # if labels do not start from 0, subtract the minimum value from all labels
        if labels.min() != 0:
            labels = labels - labels.min()
        # if the labels are not consecutive, renumber them
        if labels.max() != len(labels) - 1:
            labels = labels.rank(method='dense').astype(int) - 1
        Y.append(labels.to_numpy())

    return X, Y


def normalize(X_train, X_test, method='minmax'):
    if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise ValueError("Inputs should be numpy arrays.")

    if not isinstance(method, str) or method not in ['minmax', 'standard', 'normalize']:
        raise ValueError("Method should be either 'minmax', 'standard', or 'normalize'.")

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = Normalizer()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def convert_to_labels(predictions):
    if not isinstance(predictions, np.ndarray) or not (predictions >= 0).all() or not (predictions <= 1).all():
        raise ValueError("Predictions should be a numpy array of floats between 0 and 1.")

    labels = np.zeros(len(predictions))
    for i in range(len(predictions)):
        if predictions.shape[1] > 1:
            labels[i] = np.argmax(predictions[i])
        else:
            labels[i] = np.round(predictions[i])
    return labels.astype(int)


def categorical_to_numerical(dataframe, column):
    # convert the column to a categorical feature
    dataframe[column] = pd.Categorical(dataframe[column])

    # convert the categorical feature to a numerical feature
    dataframe[column] = dataframe[column].cat.codes

    return dataframe


def shuffle(X, Y):
    if not isinstance(X, np.ndarray) or not isinstance(Y, list) or not isinstance(Y[0], np.ndarray):
        raise ValueError("Inputs should be numpy arrays.")

    if len(X) != len(Y[0]):
        raise ValueError("The number of samples in X and Y should be the same.")

    # shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    for i in range(len(Y)):
        Y[i] = Y[i][indices]

    return X, Y


def exact_match_ratio(predictions, test_label_sets):
    num_correct = 0
    for i in range(len(predictions)):
        num_correct += sum(predictions[i] == test_label_sets[i])
    return num_correct / (len(predictions[0]) * len(predictions))


def hamming_loss(predictions, test_label_sets):
    num_wrong = 0
    for i in range(len(predictions)):
        num_wrong += sum(predictions[i] != test_label_sets[i])
    return num_wrong / (len(predictions[0]) * len(predictions))
