import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from models.name import get_experiment as ex_name
from models.description import get_experiment as ex_desc
from models.categorical import get_experiment as ex_cate


def _proccess_pred(p):
    return np.expm1(np.clip(p, 0, None))


def get_experiment(train_df, test_df):
    fit_params = {}
    indexes = np.random.randint(0, train_df.shape[0], int(train_df.shape[0]*0.2))
    first_step_train = train_df.iloc[indexes]
    second_step_train = train_df[~train_df.index.isin(indexes)]
    name_train, name_test, name_pipes, name_params, _ = ex_name(first_step_train, test_df)
    desc_train, desc_test, desc_pipes, desc_params, _ = ex_desc(first_step_train, test_df)
    cate_train, cate_test, cate_pipes, cate_params, _ = ex_cate(first_step_train, test_df)

    name_pipe = Pipeline(name_pipes)
    desc_pipe = Pipeline(desc_pipes)
    cate_pipe = Pipeline(cate_pipes)
    name_pipe.fit(name_train, **name_params)
    desc_pipe.fit(desc_train, **desc_params)
    cate_pipe.fit(cate_train, **cate_params)    
