import pandas as pd
import numpy as np
import os
import sys
import scipy
from src.config import config
from src import preprocess
import warnings

project_dir = config.PROJECT_DIR


def read_mat_file(file_path):
    """
    Read a mat file and return a dictionary of the variables.
    """
    mat_dict = scipy.io.loadmat(file_path)
    try:
        mat_dict["force_value"] = mat_dict["force_value"].squeeze()
    except:
        mat_dict["fatigue_induced"] = mat_dict["fatigue_induced"].squeeze()
    mat_dict["MVC_value"] = mat_dict["MVC_value"].squeeze()
    return mat_dict


def read_csv_file(file_path):
    """
    Read a csv file and return a pandas dataframe.
    """
    df = pd.read_csv(file_path, header=None, dtype=float).to_numpy(dtype=np.float32)
    return df

def read_trial(trial: str):
    """
    Read the emg and force csv for giving trial
    :param trial: the trial number, for example '1-1-1'
    :return: (emg, force) of given trial.
    """
    subject = f"S{trial.split('-')[0]}"
    emg_path = os.path.join(project_dir, 'data', 'processed', subject, f"{trial}-emg.csv")
    force_path = os.path.join(project_dir, 'data', 'processed', subject, f"{trial}-force.csv")
    emg = read_csv_file(emg_path)
    force = read_csv_file(force_path)
    return emg, force

def read_concentrated(filename):
    """
    Read the emg and force data from a concentrated file.
    """
    emg_path = os.path.join(project_dir, 'data', 'cache' f'emg-{filename}.csv')
    force_path = os.path.join(project_dir, 'data', 'cache' f'force-{filename}.csv')
    if os.path.exists(emg_path) and os.path.exists(force_path):
        pass
    else:
        preprocess.concentrate_data(filename)
    emg = read_csv_file(emg_path)
    force = read_csv_file(force_path)
    return emg, force


def load_data_from_cache(filename):
    """
    Load the data from the cache.
    """
    if isinstance(filename, str):
        emg_path = os.path.join(project_dir, 'data', 'cache', f'emg-{filename}.csv')
        force_path = os.path.join(project_dir, 'data', 'cache', f'force-{filename}.csv')
        if os.path.exists(emg_path) and os.path.exists(force_path):
            emg = read_csv_file(emg_path)
            force = read_csv_file(force_path)
        else:
            warnings.warn(f"Data for {filename} not found in cache. Use load_data_from_file instead.")
            emg, force = load_data_from_file(filename)
        return emg, force
    elif isinstance(filename, list):
        for subfile in filename:
            emg, force = load_data_from_cache(subfile)
            if subfile == filename[0]:
                EMG = emg
                FORCE = force
            else:
                EMG = np.vstack((EMG, emg))
                FORCE = np.vstack((FORCE, force))
        return EMG, FORCE


def load_data_from_file(filename):
    """
    Load the data from the file.
    """
    if isinstance(filename, str):
        emg, force = preprocess.concentrate_data(filename)
        return emg, force
    elif isinstance(filename, list):
        for subfile in filename:
            emg, force = load_data_from_file(subfile)
            if subfile == filename[0]:
                EMG = emg
                FORCE = force
            else:
                EMG = np.vstack((EMG, emg))
                FORCE = np.vstack((FORCE, force))
        return EMG, FORCE


def load_feature_from_cache(filename, window_size, stride):
    FILENAME = f"features-{window_size}-{stride}-{filename}"
    filepath = os.path.join(project_dir, 'data', 'cache', f'{FILENAME}.csv')
    if os.path.exists(filepath):
        FEATURES = read_csv_file(filepath)
    else:
        warnings.warn(f"Features file for {FILENAME} not found in cache.")
        subfiles = preprocess.parse_filenames(filename)
        for subfile in subfiles:
            subfilepath = os.path.join(project_dir, 'data', 'cache', f'features-{window_size}-{stride}-{subfile}.csv')
            if not os.path.exists(subfilepath):
                preprocess.calculate_features(subfile, window_size, stride)

            if subfile == subfiles[0]:
                FEATURES = read_csv_file(subfilepath)
            else:
                FEATURES = np.vstack((FEATURES, read_csv_file(subfilepath)))
        np.savetxt(filepath, FEATURES, delimiter=',')
    return FEATURES


def load_feature_from_file(filename, window_size, stride):
    subfiles = preprocess.parse_filenames(filename)
    for subfile in subfiles:
        features = preprocess.calculate_features(subfile, window_size, stride)
        if subfile == subfiles[0]:
            FEATURES = features
        else:
            FEATURES = np.vstack((FEATURES, features))
    return FEATURES


def read_subjects_info():
    """
    Read the subjects information from the file.
    """
    subjects_info_path = os.path.join(project_dir, 'data', 'subjects_info.txt')
    subjects_info = pd.read_csv(subjects_info_path, sep='\t', index_col="Label")
    return subjects_info