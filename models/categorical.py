from catboost import CatBoostRegressor


def get_experiment(train_df, test_df):
    cat_features_names = ['item_condition_id', 'category_name', 'brand_name', 'shipping']
    pipe = [
        ('cat', CatBoostRegressor(loss_function='RMSE', train_dir='./temp'))
    ]
    return train_df[cat_features_names], test_df[cat_features_names], pipe, {'cat__cat_features': [0, 1, 2, 3]}