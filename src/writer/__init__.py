import pandas as pd
import os



def write_to_results(path, subject_id, dict_data):
    """
    Write the results to a csv file`
    :param path:
    :param dict_data: dict with two keys
    :return:
    """
    if type(subject_id) == int:
        subject_id = f"S{subject_id}"
    if not os.path.exists(path):
        # Create a DataFrame with appropriate headers and save it
        columns = [subject_id]
        index = list(dict_data.keys())
        df = pd.DataFrame(columns=columns, index=index)
        df.to_csv(path)

    df = pd.read_csv(path, index_col=0)
    for key, value in dict_data.items():
        df.at[key, subject_id] = f"{value:.4f}"
    df.to_csv(path)
    return True


def save_unique_figure(fig, save_path):
    counter = 1
    filename = os.path.basename(save_path)
    basename, ext = os.path.splitext(filename)
    while os.path.exists(save_path):
        filename = f"{basename}_{counter}{ext}"
        save_path = os.path.join(os.path.dirname(save_path), filename)
        counter += 1
    fig.savefig(save_path, dpi=300)
