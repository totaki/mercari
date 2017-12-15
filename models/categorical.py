"""
Model for using predict on category, brand, shipping and condition id
"""
from utils import get_hash_params
from catboost import CatBoostRegressor


def get_experiment(train_df, test_df):
    fit_params = {'cat__cat_features': [0, 1, 2, 3]}
    cat_features_names = ['item_condition_id', 'category_name', 'brand_name', 'shipping']
    model = CatBoostRegressor(loss_function='RMSE', train_dir='./temp')
    pipe = [
        ('cat', model)
    ]
    return (
        train_df[cat_features_names],
        test_df[cat_features_names],
        pipe,
        fit_params,
        get_hash_params(model.get_params(), {}, __doc__)
    )