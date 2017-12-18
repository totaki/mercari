"""
Model for using predict on item name
"""
from models.base import Experiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor


def get_experiment():
    pipes = [
        ('cv', TfidfVectorizer(stop_words='english', max_df=0.6, norm='l2')),
        ('sgdr', SGDRegressor(max_iter=50, random_state=42))
    ]
    return Experiment(df_process=lambda df: df['name'], pipes=pipes, desctiption=__doc__)
