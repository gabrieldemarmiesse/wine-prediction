import numpy as np
import matplotlib.pyplot as plt
from functions import *

white_wine_features_target = np.genfromtxt("winequality-white-header-removed.csv", dtype=float, delimiter=';')
red_wine_features_target = np.genfromtxt("winequality-red-header-removed.csv", dtype=float, delimiter=';')

row_sums = np.concatenate((white_wine_features_target[:, :-1].max(axis=0), [1]))
white_wine_features_target = white_wine_features_target / row_sums[np.newaxis, :]

row_sums = np.concatenate((red_wine_features_target[:, :-1].max(axis=0), [1]))
red_wine_features_target = red_wine_features_target / row_sums[np.newaxis, :]

np.random.shuffle(white_wine_features_target)
num_entries = white_wine_features_target.shape[0]

train = white_wine_features_target[:int(num_entries * 0.7)]
test = white_wine_features_target[int(num_entries * 0.7):]

A = train_regressor(train)

predictions, expectations = predict_regressor(test, A)

mse_ = mse(predictions, expectations)

variance_ = variance(expectations)

functions = [lambda x: x * x, lambda x: x * x * x, lambda x: x * x * x * x]

A = train_regressor(train, functions)
predictions, expectations = predict_regressor(train, A, functions)

#find_best_lambda(white_wine_features_target, functions)


gaussians, counts = naive_bayes_train(train)

predictions, targets = naive_bayes_predict(test, gaussians, counts)

mse_ = mse(predictions, targets)

accuracy = get_classification_accuracy(predictions, targets)

confusion_matrix = get_confusion_matrix(predictions, targets)
print(confusion_matrix)

eigenvalues, eigenvectors = pca(train)

yolo = change_space(train, eigenvectors)

print(pd.DataFrame(yolo).describe())


fieozjfie=65