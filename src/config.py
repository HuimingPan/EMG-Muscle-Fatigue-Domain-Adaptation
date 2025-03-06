import yaml
import torch

def flat_dict(d):
    """
    Flatten a dictionary.
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flat_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)

class Config:
    def __init__(self, filepath=None):
        if filepath is not None:
            self.load(filepath)
            if torch.cuda.is_available():
                self.CUDA = True
            else:
                self.CUDA = False
        else:
            self.config = {}

    def __getattr__(self, item):
        try:
            return self.config[item]
        except KeyError:
            return self.config[item.lower()]

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __repr__(self):
        return str(self.config)

    def __str__(self):
        return str(self.config)

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.config, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config = flat_dict(self.config)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.config[key] = value
        return self


config = Config('../config.yml')
