import matplotlib.pyplot as plt
from src.statistics.distribution_functions import empirical_distribution_function as edf
from src.statistics.distribution_functions import probability_density_function as pdf
def plot_empirical_distribution(data):
    """
    Visualize the empirical distribution function (EDF) of a 1D array 'data'.
    :param data:
    :return: the figure object
    """
    sorted_data, data_edf = edf(data)
    fig, ax = plt.subplots()
    ax.plot(sorted_data, data_edf)
    ax.set_xlabel("Data")
    ax.set_ylabel("EDF")
    plt.show()
    return fig

def plot_probability_density(data, bins=100):
    """
    Visualize the probability density function (PDF) of a 1D array 'data'.
    :param data:
    :param bins:
    :return: the figure object
    """
    fig, ax = plt.subplots()
    bin_edges, data_pdf = pdf(data, bins=bins)
    ax.hist(data, bins=bins, density=True, alpha=0.5)
    ax.set_xlabel("Data")
    ax.set_ylabel("PDF")
    plt.show()
    return bin_edges, data_pdf