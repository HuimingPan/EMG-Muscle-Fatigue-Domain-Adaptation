import numpy as np
import os
from src.config import config
from src import reader
from src.feature.feature_extraction import extract_features

project_dir = config.PROJECT_DIR


def get_trigger_index(trigger_channel) -> int:
    """
    Get the start index of the EMG signal according to the trigger channel.
    """
    slope = np.diff(trigger_channel)
    trigger_index = np.nonzero(slope > 10)[0]
    second_trigger = trigger_index[np.nonzero(np.diff(trigger_index) > 2048)[0]+1]
    print(f"Potential trigger index: {trigger_index}")
    if second_trigger.any():
        triggers = [trigger_index[0], second_trigger[0]]
    else:
        triggers = trigger_index[0]
    print(f"Trigger index: {triggers}")
    return triggers

def get_trigger_channel(emg: np.array, channel: int = 192):
    """
    Get the trigger channel from the raw EMG data.
    """
    return emg[:, channel]


def normalize_force(force_dict: dict) -> np.array:
    """
    Normalize the force signal.
    """
    try:
        force = force_dict["force_value"]
    except:
        force = force_dict["fatigue_induced"]
    force = force / force_dict["MVC_value"]
    return force


def concentrate_data(filename: str):
    """
    Concentrate the data from the different trials into one file.
    """
    # Check the number of arguments and parse them.
    filenames = parse_filenames(filename)
    for subfilename in filenames:
        # Load data
        subject, session, trial = subfilename.split("-")
        emg_data = reader.read_csv_file(
            os.path.join(project_dir, 'data', 'processed', f'S{subject}', subfilename + "-emg.csv"))
        force_data = reader.read_csv_file(
            os.path.join(project_dir, 'data', 'processed', f'S{subject}', subfilename + "-force.csv"))

        # Concentrate data
        if subfilename == filenames[0]:
            emg = emg_data
            force = force_data
        else:
            emg = np.concatenate((emg, emg_data), axis=0)
            force = np.concatenate((force, force_data), axis=0)

    # Save concentrated data
    saved_path = os.path.join(project_dir, 'data', 'cache',
                              f'emg-{filename}.csv')
    np.savetxt(saved_path, emg, delimiter=',')
    saved_path = os.path.join(project_dir, 'data', 'cache',
                              f'force-{filename}.csv')
    np.savetxt(saved_path, force, delimiter=',')

    return emg, force


def parse_filenames(filename: str) -> tuple:
    """
    Parse the filename of the concentrated data and return the comprised filenames.
    """
    args = filename.split("-")
    if len(args) not in [3, 5]:
        raise ValueError("The number of arguments must be 3 or 5.")
    elif len(args) == 3:
        subjects = args[0]
        sessions = args[1]
        trials = args[2]
        filenames = []
        for subject in subjects:
            for session in sessions:
                for trial in trials:
                    filenames.append(f"{subject}-{session}-{trial}")
    else:  # len(args) == 5, subject-session-trial-session2-trial2
        subjects = [args[0]]
        sessions = [args[1], args[3]]
        trials = [args[2], args[4]]
        filenames = []
        for subject in subjects:
            for session in sessions[0]:
                for trial in trials[0]:
                    filenames.append(f"{subject}-{session}-{trial}")
            for session in sessions[1]:
                for trial in trials[1]:
                    filenames.append(f"{subject}-{session}-{trial}")
    return filenames


def calculate_features(filename, window_size, stride):
    subject = filename.split("-")[0]
    emg_data = reader.read_csv_file(
        os.path.join(project_dir, 'data', 'processed', f'S{subject}', filename + "-emg.csv"))
    force_data = reader.read_csv_file(
        os.path.join(project_dir, 'data', 'processed', f'S{subject}', filename + "-force.csv"))

    for start_idx in range(0, emg_data.shape[0] - window_size, stride):
        end_idx = start_idx + window_size
        data_win = emg_data[start_idx:end_idx, :]
        feature = extract_features(data_win)
        force_idx = int(end_idx / config.EMG_FS * config.FORCE_FS)
        feature = np.hstack((feature, force_data[force_idx]))
        if start_idx == 0:
            features = feature
        else:
            features = np.vstack((features, feature))
    savepath = os.path.join(project_dir, 'data', 'cache', f'features-{window_size}-{stride}-{filename}.csv')
    np.savetxt(savepath, features, delimiter=",")
    return features
