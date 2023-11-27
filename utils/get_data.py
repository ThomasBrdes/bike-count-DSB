import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def _read_data(path, f_name):
    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

def _merge_external_data_weather(X, external_data_path):

    # We conserve the original index after the merge
    X = X.copy()
    X["orig_index"] = np.arange(X.shape[0])
    
    # Read external data and transform 
    df_ext = pd.read_csv(external_data_path, parse_dates=["date"])
    
    # We use pd.to_datetime() function to standardize the date resolution to micro seconds
    X["date"] = pd.to_datetime(X["date"])
    df_ext["date"] = pd.to_datetime(df_ext["date"]).astype('datetime64[us]')
    
    # Left join merge 
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t"]].sort_values("date"), on="date"
    )
    
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    
    return X

def get_train_data(path="."):
    f_name = "train.parquet"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.parquet"
    return _read_data(path, f_name)

def read_original_data():

    X_train, y_train = problem.get_train_data()
    X_test, y_test = problem.get_test_data()

    return X_train, y_train, X_test, y_test
