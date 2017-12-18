import hashlib
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline


def get_splited_categories(df):
    splited_categories  = [
       c.split('/') if c != NULL else []
       for c in df['category_name'].values
    ]
    max_category_level_deep = max([len(c) for c in splited_categories])
    for s in splited_categories:
        s.extend([NULL] * (max_category_level_deep - len(s)))
    return splited_categories


def create_small_dataset(input_fn, output_prefix, max_train, max_test):
    import pandas as pd
    data = pd.read_table(input_fn)
    train_indexes = np.random.randint(data.shape[0], size=max_train)
    test_indexes = np.random.randint(data.shape[0], size=max_test)
    train_df = data.iloc[train_indexes]
    test_df = data.iloc[train_indexes]
    train_df.to_csv('./data/%s_train.csv' % output_prefix, index=False)
    test_df.to_csv('./data/%s_test.csv' % output_prefix, index=False)


def calcs_msle(y, p):
    return mean_squared_log_error(y, np.clip(p, 0, None))


def calc_metrics(x, y, model):
    predictions = model.predict(x)
    return calcs_msle(y, np.expm1(predictions))

def print_dict(dct):
    for k, v in dct.items():
        print('  %s: ' % k, v)


def get_hash_params(*args):
    string = ''.join([str(a) for a in args])
    md5 = hashlib.md5()
    md5.update(string.encode())
    return md5.hexdigest()        


def run(experiment, X_train, Y_train, X_test, Y_test):
    pipe = experiment.model
    print('\n')
    for _, model in pipe.named_steps.items():
        print('Model: ', model.__class__.__name__)
        print_dict(model.get_params())
        print('\n')
    print('Fit params:')
    print_dict(experiment.fit_params)
    print('\n')
    
    print('Run:')
    print('  fiting')
    pipe.fit(experiment.convert_dfs(X_train), Y_train, **experiment.fit_params)
    print('  testing')
    metrics = calc_metrics(experiment.convert_dfs(X_test), Y_test, pipe)
    print('\n')
    
    print('Metrics:')
    print('  %s\n' % metrics)
    return metrics, pipe