import numpy as np
import scipy


# Extract features of a window of shape (n_samples, n_channels)
# Time domain features: mean, std, median, mean abosulte value, waveform length,  range, root mean square, skewness, kurtosis,

def _mean(window):
    return np.mean(window, axis=0)


def _std(window):
    return np.std(window, axis=0)


def _median(window):
    return np.median(window, axis=0)


def _mean_absolute_value(window):
    return np.mean(np.abs(window), axis=0)


def _waveform_length(window):
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def _zero_crossing_rate(window):
    return np.sum(np.diff(np.sign(window), axis=0), axis=0)


def _slope_sign_change(window):
    return np.sum(np.abs(np.diff(np.sign(np.diff(window, axis=0)), axis=0)), axis=0) / 2


def _root_mean_square(window):
    return np.sqrt(np.mean(window ** 2, axis=0))


def _skewness(window):
    return scipy.stats.skew(window, axis=0)


def _kurtosis(window):
    return scipy.stats.kurtosis(window, axis=0)


def _median_absolute_deviation(window):
    return np.median(np.abs(window - np.median(window, axis=0)), axis=0)


def _interquartile_range(window):
    return scipy.stats.iqr(window, axis=0)


def _waveform_length(window):
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)

def _spectral_peak(data_win,  win_size):
    N = data_win.shape[0]  # N=win_size
    mean = np.mean(data_win, axis=0)
    data_win = data_win - mean
    spectral = fft(data_win, axis=0)
    abs_spectral = np.abs(spectral) / N
    abs_spectral_half = abs_spectral[:win_size // 2, :]
    index_spectral_peak = np.argmax(abs_spectral_half, axis=0)
    spectral_peak_frequency = np.max(abs_spectral_half, axis=0)

    return index_spectral_peak, spectral_peak_frequency
def extract_features(window):
    mean = _mean(window)
    std = _std(window)
    median = _median(window)
    mean_absolute_value = _mean_absolute_value(window)
    zero_crossing_rate = _zero_crossing_rate(window)
    slope_sign_change = _slope_sign_change(window)
    root_mean_square = _root_mean_square(window)
    skewness = _skewness(window)
    kurtosis = _kurtosis(window)
    median_absolute_deviation = _median_absolute_deviation(window)
    interquartile_range = _interquartile_range(window)
    waveform_length = _waveform_length(window)
    index_spectral_peak, spectral_peak_frequency = __spectral_peak(data_win, win_size)
    feature = np.concatenate([mean, std, median, mean_absolute_value, waveform_length,
                              zero_crossing_rate, slope_sign_change, root_mean_square, skewness, kurtosis,
                              median_absolute_deviation, interquartile_range, index_spectral_peak, spectral_peak_frequency])
    return feature


if __name__ == '__main__':
    window = np.random.rand(100, 8)
    feature = extract_features(window)
    print(feature)
    print(feature.shape)
