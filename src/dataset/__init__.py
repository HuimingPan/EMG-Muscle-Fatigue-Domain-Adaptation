from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset
from src import reader
from src.registry import DATASETS


class BaseDataset(Dataset, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.trials = cfg.TRIALS
        self.config = cfg
        self.electrodes = list(range(0, 192)) if cfg.ELECTRODES == "All" else (
                cfg.ELECTRODES + [e + 64 for e in cfg.ELECTRODES] + [e + 128 for e in cfg.ELECTRODES])
        self.window_size = cfg.WINDOW_SIZE
        self.stride = cfg.STRIDE

        self.emg_fs = cfg.EMG_FS
        self.force_fs = cfg.FORCE_FS
        self.device = cfg.DEVICE
        self.start_idx = None
        self.X = None
        self.y = None
        self.data_list = []
        self.indices = []

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def to_device(self, device='cuda'):
        if isinstance(self.X, np.ndarray):
            self.X = torch.tensor(self.X, device=device, dtype=torch.float)
            self.y = torch.tensor(self.y, device=device, dtype=torch.float)
        elif isinstance(self.X, torch.Tensor):
            self.X = self.X.to(device)
            self.y = self.y.to(device)


@DATASETS.register()
class Dataset1D(BaseDataset):
    def __init__(self, cfg):
        super(Dataset1D, self).__init__(cfg)
        self.load_data()

    def load_data(self):
        """
        Load data from trials
        """
        data_list = []
        INDICES = []
        for i, trial in enumerate(self.trials):
            emg, force = reader.read_trial(trial)
            emg = torch.tensor(emg, dtype=torch.float32).to(self.device)
            force = torch.tensor(force, dtype=torch.float32).squeeze().to(self.device)

            indices = np.arange(0, emg.shape[0] - self.window_size, self.stride)
            fatigue_status = 0 if trial.split("-")[1] == "1" else 1
            data_list.append((emg, force, fatigue_status))
            INDICES.append((np.ones_like(indices) * i, indices))

        self.data_list = data_list
        self.indices = np.hstack(INDICES).T

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        try:
            i, index = self.indices[item]
            force_idx = int((index + self.window_size) / self.emg_fs * self.force_fs)
            emg = self.data_list[i][0][index:index + self.window_size, :]
            force = self.data_list[i][1][force_idx]
            fatigue_status = self.data_list[i][2]
        except:
            print(item, i, index)
            raise ValueError("Index Error")
        return emg.T, force, fatigue_status


@DATASETS.register()
class DatasetFeature(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        if self.use_cache:
            features = reader.load_feature_from_cache(self.filename, self.window_size, self.stride)
        else:
            features = reader.load_feature_from_file(self.filename, self.window_size, self.stride)
        self.X = features[:, :-1]
        self.y = features[:, -1]

    def __getitem__(self, item):
        return self.X[item, :].T, self.y[item]

    def __len__(self):
        return len(self.y)


@DATASETS.register()
class Dataset_Mean(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        if self.use_cache:
            self.X, self.y = reader.load_data_from_cache(self.filename)
        else:
            self.X, self.y = reader.load_data_from_file(self.filename)
        self.y = self.y.squeeze()
        self.start_idx = np.arange(0, self.X.shape[0] - self.window_size, self.stride)
        self.X_ = np.zeros((len(self.start_idx), self.X.shape[1]))
        self.y_ = np.zeros(len(self.start_idx))
        for i in range(len(self.start_idx)):
            idx = self.start_idx[i]
            end_idx = idx + self.window_size
            force_idx = int(end_idx / self.emg_fs * self.force_fs)
            window_data = self.X[idx:end_idx]
            self.X_[i] = np.mean(window_data, axis=0)
            self.y_[i] = self.y[force_idx]
        self.X = self.X_
        self.y = self.y_

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.start_idx)


@DATASETS.register()
class DatasetSingleFrame(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        if self.use_cache:
            self.X, self.y = reader.load_data_from_cache(self.filename)
        else:
            self.X, self.y = reader.load_data_from_file(self.filename)
        self.y = self.y.squeeze()

    def __getitem__(self, item):
        emg_idx = int(item / self.force_fs * self.emg_fs)
        return self.X[emg_idx, :], self.y[item]

    def __len__(self):
        return len(self.y)


class Dataset2D(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        """
        Load data from trials
        """
        data_list = []
        INDICES = []
        for i, trial in enumerate(self.trials):
            emg, force = reader.read_trial(trial)
            emg = torch.tensor(emg, dtype=torch.float32)
            force = torch.tensor(force, dtype=torch.float32).squeeze()

            emg = emg.reshape(emg.shape[0], 8, -1)

            indices = np.arange(0, emg.shape[0] - self.window_size, self.stride)
            fatigue_status = 0 if trial.split("-")[1] == "1" else 1
            data_list.append((emg, force, fatigue_status))
            INDICES.append((np.ones_like(indices) * i, indices))

        self.data_list = data_list
        self.indices = np.hstack(INDICES).T

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        try:
            i, index = self.indices[item]
            force_idx = int((index + self.window_size) / self.emg_fs * self.force_fs)
            emg = self.data_list[i][0][index:index + self.window_size, :, :]
            force = self.data_list[i][1][force_idx]
            fatigue_status = self.data_list[i][2]
        except:
            print(item, i, index)
            raise ValueError("Index Error")
        return emg, force, fatigue_status


class AdverDataset(BaseDataset):
    """
    New Dataset
    """

    def __init__(self, cfg):
        super(AdverDataset, self).__init__(cfg)
        self.load_data()

    def load_data(self):
        """
        Load data from trials
        """
        data_list = []
        INDICES = []
        for i, trial in enumerate(self.trials):
            emg, force = reader.read_trial(trial)
            emg = torch.tensor(emg, dtype=torch.float32)
            force = torch.tensor(force, dtype=torch.float32).squeeze()

            indices = np.arange(0, emg.shape[0] - self.window_size, self.stride)

            if trial.split("-")[1] == "1":
                fatigue_status = 0
            else:
                fatigue_status = 1

            data_list.append((emg, force, fatigue_status))
            INDICES.append((np.ones_like(indices) * i, indices))

        self.data_list = data_list
        self.indices = np.hstack(INDICES).T

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        try:
            i, index = self.indices[item]
            force_idx = int((index + self.window_size) / self.emg_fs * self.force_fs)
            emg = self.data_list[i][0][index:index + self.window_size, self.electrodes]
            force = self.data_list[i][1][force_idx]
            fatigue_status = self.data_list[i][2]
        except IndexError:
            print(item, i, index)
            raise ValueError("Index Error")
        return emg.T, force, fatigue_status
