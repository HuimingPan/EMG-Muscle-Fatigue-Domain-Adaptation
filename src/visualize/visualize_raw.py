import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animation(emg):
    """
    Create an animation of the EMG signal.
    :param emg:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.matshow(row2pattern(emg[0, :]), aspect='auto', cmap='viridis')
    plt.colorbar(im)

    def init():
        ax.set_title("EMG signal: 0")
        im.set_data(row2pattern(emg[0, :]))
        return [im]

    def update(i):
        ax.set_title(f"EMG signal: {i / 1000}")
        im.set_data(row2pattern(emg[i, :]))
        return [im]

    ani = FuncAnimation(fig, update, frames=emg.shape[0], init_func=init, blit=True)
    plt.show()

def row2pattern(row):
    p1 = row[:64].reshape(8,8).T
    p2 = row[64:128].reshape(8,8).T
    p3 = row[128:192].reshape(8,8).T
    return np.concatenate((p2,p1,p3), axis=1)