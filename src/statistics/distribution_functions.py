import numpy as np


def empirical_distribution_function(data):
    """
    Calculate the empirical distribution function (EDF) of a 1D array 'data'.
    :param data: 1D array of data
    :return: tuple of sorted data and EDF
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    edf = [(i + 1) / n for i in range(n)]
    return sorted_data, edf

def probability_density_function(data, bins=10):
    """
    Calculate the probability density function (PDF) of a 1D array 'data'.
    :param data: 1D array of data
    :param bins: number of bins to use for the histogram
    :return: tuple of bin edges and PDF
    """
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    pdf = hist / np.sum(hist)
    return bin_edges, pdf
