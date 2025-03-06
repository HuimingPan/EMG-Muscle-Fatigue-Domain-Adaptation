import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from decimal import Decimal
import os


def cm_plot(target, pred, label_dict={"NF": 0, "F": 1}, title=None):
    """
    Plot the confusion matrix
    :param target:
    :param pred:
    :param label_dict: The dictionary of the label, e.g. {"NF": 0, "F": 1}
    :param title:
    :return:
    """
    if type(target) is list:
        target = np.array(target).reshape(-1, 1)
    if type(pred) is list:
        pred = np.array(pred).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    cm = metrics.confusion_matrix(target, pred, normalize='true')

    labels = list(label_dict.keys())
    ax.matshow(cm, cmap=plt.cm.Blues)
    num_local = np.array(range(len(labels)))
    for x in range(len(cm)):
        for target in range(len(cm)):
            if x == target:
                plt.text(x, target, str(Decimal(cm[target, x] * 100).quantize(Decimal('0.0'))),
                         horizontalalignment='center',
                         verticalalignment='center', color='black')
            else:
                plt.text(x, target, str(Decimal(cm[target, x] * 100).quantize(Decimal('0.0'))),
                         horizontalalignment='center',
                         verticalalignment='center', color='black')
    ax.set_yticks(num_local, labels)  # 将标签印在y轴坐标上
    ax.set_xticks(num_local, labels)  # 将标签印在x轴坐标上
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Predicted label')  # 坐标轴标签
    ax.set_ylabel('True label')  # 坐标轴标签
    fig.suptitle(title) if title else None
    return fig


def plot_confusion_matrix(*args, label_dict={"NF": 0, "F": 1}, **kwargs):
    """
    Plot the confusion matrix
    :param args: arguments can be either target and pred or a dataframe
    :param label_dict:
    :param kwargs:
    :return:
    """
    if len(args) == 2:
        target, pred = args
        cm = metrics.confusion_matrix(target, pred, normalize='true')

    elif len(args) == 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]
        subjects = df["Subject"].unique()
        cm = {}
        acc = []
        for subject in subjects:
            target = df[df["Subject"] == subject]["Target"]
            pred = df[df["Subject"] == subject]["Predicted"]
            cm[subject] = metrics.confusion_matrix(target, pred, normalize='true') * 100
            acc.append(metrics.accuracy_score(target, pred) * 100)
        std = np.std(acc)
        acc = np.mean(acc)
        print(f"Average accuracy: {acc:.2f}+- {std: .2f}%")
        fig, ax = plt.subplots()
        mean_cm = np.mean([cm[subject] for subject in subjects], axis=0)
        std = np.std([cm[subject] for subject in subjects], axis=0)
        ax.matshow(mean_cm, cmap=plt.cm.Blues)
        num_local = np.array(range(len(mean_cm)))
        for x in range(len(mean_cm)):
            for target in range(len(mean_cm)):
                color = "white" if mean_cm[target, x] > 50 else "black"
                if x == target:
                    plt.text(x, target, f"{mean_cm[target, x]:.2f}±{std[target, x]:.2f}",
                             horizontalalignment='center',
                             verticalalignment='center', color=color)
                else:
                    plt.text(x, target, f"{mean_cm[target, x]:.2f}±{std[target, x]:.2f}",
                             horizontalalignment='center',
                             verticalalignment='center', color=color)
        labels = list(label_dict.keys())
        ax.set_yticks(num_local, labels)
        ax.set_xticks(num_local, labels)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        plt.show()
        save_path = r"E:\OneDrive - sjtu.edu.cn\Research\Papers\EMG-based fatigue force estimation\figures\Results\confusion matrix.png"
        save_unique_figure(fig, save_path)
        return fig


def save_unique_figure(fig, save_path):
    counter = 1
    filename = os.path.basename(save_path)
    basename, ext = os.path.splitext(filename)
    while os.path.exists(save_path):
        filename = f"{basename}_{counter}{ext}"
        save_path = os.path.join(os.path.dirname(save_path), filename)
        counter += 1
    fig.savefig(save_path, dpi=300)
