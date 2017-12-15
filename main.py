import sys
import os
import hashlib
import datetime
import pandas as pd
import numpy as np
import utils


class ComplexOut():

    def __init__(self, string, force=False):
        md5 = hashlib.md5()
        md5.update(string.encode())
        self._file_name = './results/%s.txt' % md5.hexdigest()[-8:]
        self._original = sys.stdout
        
        if not os.path.exists(self._file_name) or force:
            self._file = open(self._file_name, 'w')
        else:
            self._file = None
        sys.stdout = self

    def write(self, text):
        self._original.write(text)
        if self._file:
            self._file.write(text)
    
    def print_cache(self):
        with open(self._file_name) as f:
            print(f.read())
            print('Using cache: %s' % self._file_name)

    @property
    def is_cached(self):
        return not self._file

    def revert(self):
        sys.stdout = self._original
        if self._file:
            self._file.close()
    
    def flush(self):
        self._original.flush()
    
    def delete_current_cache_item(self):
        os.remove(self._file_name)
        self._file = None


if __name__ == '__main__':
    import argparse
    import importlib
    parser = argparse.ArgumentParser(description='Process models.')
    parser.add_argument('model', choices=['description', 'name', 'categorical'])
    parser.add_argument('--force', action='store_true', help='Run experiment ignore cache')
    args = parser.parse_args()

    train_df = pd.read_csv('./data/50K_1K_R_train.csv')
    test_df = pd.read_csv('./data/50K_1K_R_test.csv')
    Y_train = np.log1p(train_df['price'].values)
    Y_test = test_df['price'].values


    def fillna():
        null = 'None'
        for i in ['category_name', 'brand_name', 'item_description']:
            train_df[i].fillna(null, inplace=True)
            test_df[i].fillna(null, inplace=True)

    fillna()

    experiment_module = importlib.import_module('models.%s' % args.model)
    np.random.seed(42)
    X_train, X_test, pipes, fit_params = experiment_module.get_experiment(train_df, test_df)
    
    string_experiment = '%s%s%s' % (pipes, fit_params, args.model)

    complex_out = ComplexOut(string_experiment, force=args.force) 
    if complex_out.is_cached:
        complex_out.print_cache()
    else:
        try:
            print(datetime.datetime.utcnow().isoformat())
            metrics, pipe = utils.run(pipes, X_train, Y_train, X_test, Y_test, fit_params=fit_params)
        except KeyboardInterrupt:
            complex_out.delete_current_cache_item()
            print('\nForce stop from KeyboardInterrupt')
    complex_out.revert()