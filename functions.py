import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *

def features_targets(array):
    return array[:, :-1], array[:, -1:]


def extend(X, functions):
    list_of_new_values = []
    for function in functions:
        new_values = function(X)
        list_of_new_values.append(new_values)
    for new_values in list_of_new_values:
        X = np.concatenate((X, new_values), axis=1)

    lines_number = X.shape[0]
    X_with_ones = np.concatenate((np.array([[1] * lines_number]).T, X), axis=1)
    return X_with_ones


def train_regressor(array, functions=[]):
    X, Y = features_targets(array)
    X = extend(X, functions)

    inverse = np.linalg.pinv(X.T.dot(X))
    A = inverse.dot(X.T).dot(Y)
    return A


def predict_regressor(array, A, functions=[]):
    new_X, Y = features_targets(array)
    new_X = extend(new_X, functions)
    return new_X.dot(A), Y


def mse(predictions, expected):
    size = predictions.shape[0]
    errors = predictions - expected
    error = errors.T.dot(errors)
    return error[0, 0] / size


def variance(values):
    size = values.shape[0]
    ones = np.array([[1] * size])
    mean = (ones.dot(values) / size)[0, 0]
    distance_to_mean = ones.T * mean - values
    variance = (distance_to_mean.T.dot(distance_to_mean) / size)[0, 0]
    return variance


def train_regularized_regressor(array, functions=[], lambda_=0):
    X, Y = features_targets(array)
    X = extend(X, functions)

    I = np.identity(X.shape[1])
    I[0, 0] = 0

    inverse = np.linalg.pinv(X.T.dot(X) + lambda_ * I)
    A = inverse.dot(X.T).dot(Y)
    return A


def cross_validation(data, train_function, prediction_function, nb_slices):
    np.random.shuffle(data)
    intervals = np.linspace(0, len(data), nb_slices)
    indexes = [int(i) for i in intervals]

    mse_list = []
    for i in range(nb_slices - 1):
        test = data[indexes[i]:indexes[i + 1]]
        train = np.concatenate((data[:indexes[i]], data[indexes[i + 1]:]), axis=0)
        A = train_function(train)
        prediction, targets = prediction_function(test, A)
        current_mse = mse(prediction, targets)
        mse_list.append(current_mse)

    mean = sum(mse_list) / len(mse_list)
    return mean


def find_best_lambda(train, test, functions=[]):
    extreme = [0, 10]
    nb_points = 100
    mse_list = []
    for lambda_ in np.linspace(extreme[0], extreme[1], nb_points):
        A = train_regularized_regressor(train, functions, lambda_=lambda_)
        predictions, Y_test = predict_regressor(test, A)
        mse_list.append(mse(predictions, Y_test))

        # plt.plot(np.linspace(extreme[0],extreme[1],nb_points),mse_list)


def find_best_lambda_cross_validation(data, functions=[]):
    extreme = [0, 10]
    nb_points = 1000
    mse_list = []
    for lambda_ in np.linspace(extreme[0], extreme[1], nb_points):
        def regressor(x):
            return train_regularized_regressor(x, functions, lambda_)

        def predict(x, y):
            return predict_regressor(x, y, functions)

        current_mse = cross_validation(data, regressor, predict, 10)

        mse_list.append(current_mse)

    m = min(mse_list)
    idx = mse_list.index(m)
    best_lambda = np.linspace(extreme[0], extreme[1], nb_points)[idx]
    plt.plot(np.linspace(extreme[0], extreme[1], nb_points), mse_list)
    plt.show()


def naive_bayes_train(data):
    df = pd.DataFrame(data)

    # We group by the quality of the wine
    grouped = df.groupby([11])

    # Now we get the variances and means to create later the gaussian distributions
    gaussians = dict()
    counts = dict()
    for quality, set in grouped:
        means = list(set.mean())[:-1]
        variances = list(set.var())[:-1]

        counts[int(quality)] = set.shape[0]

        # We create a list of gaussian distributions
        a=[]
        for i in range(11):
            a.append((means[i], variances[i]))
        gaussians[int(quality)] = a
    return gaussians, counts


def gaussian(x, mean, variance):
    exponential = exp(-(x-mean)**2/(2*variance))

    result = exponential/(sqrt(variance * 2 * pi))

    return result



def naive_bayes_predict(data, table, counts):
    X, Y = features_targets(data)
    predictions = []
    for row in X:
        best_class_and_score = (0,0)
        for key, value in table.items():
            scores = [gaussian(row[i], value[i][0], value[i][0]) for i in range(11)]
            score = np.prod(scores) * counts[key]
            if score > best_class_and_score[1]:
                best_class_and_score = (key, score)

        predictions.append(best_class_and_score)

    return np.array([prediction[0] for prediction in predictions]).T, Y

def pca(data):
    df = pd.DataFrame(data)
    df = df - df.mean()
    matrix = df.values[:, :-1]
    cov_matrix = matrix.T.dot(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors

def change_space(data, A):
    X, Y = features_targets(data)
    df = pd.DataFrame(X)
    df = df - df.mean()
    X=df.values
    X_new = X.dot(A.T)
    new_data = np.concatenate((X_new, Y), axis = 1)
    return new_data


