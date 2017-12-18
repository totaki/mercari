from sklearn.pipeline import Pipeline
from utils import get_hash_params


class Experiment(object):

    def __init__(self, df_process, pipes, fit_params=None, desctiption=""):
        self._df_proccess = df_process
        self._pipes = pipes
        self._fit_params = fit_params if fit_params else {}
        self._description = desctiption

    def convert_dfs(self, *dfs):
        processed = [self._df_proccess(df) for df in dfs]
        if len(processed) == 1:
            processed = processed[0]
        return processed if len(processed) else None

    @property
    def model(self):
        return Pipeline(self._pipes)

    @property
    def fit_params(self):
        return self._fit_params

    @property
    def hash(self):
        return get_hash_params(self._pipes, self._fit_params, self._description)
