"""
Model for using predict on item name
"""
from utils import get_hash_params
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor


def get_experiment(train_df, test_df):
    fit_params = {}
    pipe = [
        ('cv', TfidfVectorizer(stop_words='english', max_df=0.6, norm='l2')),
        ('sgdr', SGDRegressor(max_iter=50, random_state=42))
    ]
    return (
        train_df['name'],
        test_df['name'],
        pipe,
        fit_params,
        get_hash_params(pipe, {}, __doc__)
    )