from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# import problem
# from submissions.starting_kit.estimator import _encode_dates
# from submissions.external_data.estimator import _encode_dates, _merge_external_data



def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

def _read_data(path, f_name):
    _target_column_name = "log_bike_count"
    data = pd.read_parquet(f_name)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_train_data(path="."):
    f_name = "../input/mdsb-2023/train.parquet"
    _target_column_name = "log_bike_count"
    data = pd.read_parquet(f_name)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array



def get_test_data(path="."):
    f_name = "../input/mdsb-2023/final_test.parquet"
    data = pd.read_parquet(f_name)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    return data

def read_data():

    X_train, y_train = get_train_data()
    X_test = get_test_data()

    return X_train, y_train, X_test


def preprocessing(X_train):
    
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    return preprocessor, date_encoder

    
def train_model(X_train, y_train, preprocessor, date_encoder):
    
    regressor = Ridge()

    pipe = make_pipeline(date_encoder, preprocessor, regressor)
    pipe.fit(X_train, y_train)
    
    return pipe 

    
def submission_kaggle(pipe, X_test):
    y_pred = pipe.predict(X_test)
    results = pd.DataFrame(
        dict(
            Id=np.arange(y_pred.shape[0]),
            log_bike_count=y_pred,
        )
    )
    results.to_csv("submission.csv", index=False)
    

def main():

    # Read data
    X_train, y_train, X_test = read_data()
    
    # Get preprocessor
    preprocessor, date_encoder = preprocessing(X_train)
    
    # train_model
    pipe = train_model(X_train, y_train, preprocessor, date_encoder)
    
    # Get submission kaggle to csv
    submission_kaggle(pipe, X_test)
    
    
if __name__ == "__main__":
    main()