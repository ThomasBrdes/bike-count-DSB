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
import problem
# from submissions.starting_kit.estimator import _encode_dates
from submissions.external_data.estimator import _encode_dates, _merge_external_data

def read_data():

    X_train, y_train = problem.get_train_data()
    X_test, y_test = problem.get_test_data()

    return X_train, y_train, X_test, y_test


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
    

def get_RMSE_local(pipe, X_train, y_train, X_test, y_test):
    print(
        f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}"
    )
    print(
        f"Test set, RMSE={mean_squared_error(y_test, pipe.predict(X_test), squared=False):.2f}"
    )   

    
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
    X_train, y_train, X_test, y_test = read_data()
    
    # Get preprocessor
    preprocessor, date_encoder = preprocessing(X_train)
    
    # train_model
    pipe = train_model(X_train, y_train, preprocessor, date_encoder)
    
    # Predict data and get RMSE
    get_RMSE_local(pipe, X_train, y_train, X_test, y_test)
    
    # Get submission kaggle to csv
    submission_kaggle(pipe, X_test)
    
    
if __name__ == "__main__":
    main()