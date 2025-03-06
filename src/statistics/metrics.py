import numpy as np
from sklearn import metrics


def rmse_and_cc(estimate, label):
    rmse = calculate_rmse(estimate, label)
    cc = calculate_correlation_coefficient(estimate, label)
    return rmse, cc


def calculate_rmse(estimate, label):
    return np.sqrt(metrics.mean_squared_error(label, estimate))


def calculate_correlation_coefficient(estimate, label):
    cc = np.corrcoef(estimate, label)[0, 1]
    return cc
