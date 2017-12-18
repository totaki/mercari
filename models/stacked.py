from models.name import get_experiment as ex_name
from models.description import get_experiment as ex_desc
from models.categorical import get_experiment as ex_cate
import pandas as pd
import numpy as np


def get_experiment(train_df, test_df):
    fit_params = {}
    indexes = np.random.randint(0, train_df.shape[0], int(train_df.shape[0]*0.2))
    first_step_train = train_df.iloc[indexes]
    second_step_train = train_df[~train_df.index.isin(indexes)]
    name_train, _, name_model, name_params, _ = ex_name(first_step_train, test_df)
    desc_train, _, desc_model, desc_params, _ = ex_desc(first_step_train, test_df)
    cate_train, _, cate_model, cate_params, _ = ex_cate(first_step_train, test_df)
