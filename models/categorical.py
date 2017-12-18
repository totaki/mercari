"""
Model for using predict on category, brand, shipping and condition id
"""
from utils import get_hash_params
from models.base import Experiment
from catboost import CatBoostRegressor


class CatExperiment(Experiment):

    @property
    def hash(self):
        cat_model = self._pipes[0][1]
        return get_hash_params(cat_model.get_params(), self._fit_params, __doc__)


def get_experiment():
    fit_params = {'cat__cat_features': [0, 1, 2, 3]}
    cat_features_names = ['item_condition_id', 'category_name', 'brand_name', 'shipping']
    pipes = [
        ('cat', CatBoostRegressor(loss_function='RMSE', train_dir='./temp'))
    ]
    return CatExperiment(
        df_process=lambda df: df[cat_features_names],
        pipes=pipes,
        fit_params=fit_params,
    )
