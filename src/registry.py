class Registry:
    """
    A registry to register classes and be able to instantiate them by name.

    """
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name=None):
        def wrapper(cls):
            if name is None:
                self._registry[cls.__name__] = cls
            else:
                self._registry[name] = cls
            return cls
        return wrapper

    def __getattr__(self, item):
        return self._registry[item]

    def __str__(self):
        s = f"{self._name} registry with {len(self._registry)} items\n"
        for k, v in self._registry.items():
            s += f"\t{k}: {v}\n"
        return s

    def get(self, name):
        return self._registry[name]


MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
