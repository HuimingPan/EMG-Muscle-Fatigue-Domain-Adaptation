"""

"""

import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.models.nn import BaseModel
from src.config import config
import pandas as pd
from src.writer import write_to_results
from src.dataset import BaseDataset
from src.visualize.visualize_results import plot_fitting_curve
from src.visualize.visualize_confusion_matrix import cm_plot
from src.models.reverse_layer import GradReverse
from src.models.nn import Estimator, DomainClassifier
from src.dataset import AdverDataset
from src import reader
from src.trainer import BaseTrainer

config.update(FEATURE_NUM=256)


class Trainer(BaseTrainer):
    def __init__(self, subject_num, config, Estimator, DomainClassifier, Dataset, **kwargs):
        super(Trainer, self).__init__(subject_num, config, Estimator, DomainClassifier, Dataset, **kwargs)


if __name__ == '__main__':
    for subject_num in [1, 3, 4, 6, 7, 8, 9]:
        for t in [[1, 4, 7], [2, 5, 8], [3, 6, 9]]:
            trainer = Trainer(subject_num=subject_num, config=config, Estimator=Estimator,
                              DomainClassifier=DomainClassifier, Dataset=AdverDataset,
                              test_trials=t)
            trainer.train()
            trainer.evaluation(suffix=f"k{t[0]}")
