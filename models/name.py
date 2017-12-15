from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor


def get_experiment(train_df, test_df):
    pipe = [
        ('cv', TfidfVectorizer(stop_words='english', max_df=0.6, norm='l2')),
        ('sgdr', SGDRegressor(max_iter=50, random_state=42))
    ]
    return train_df['name'], test_df['name'], pipe, {}