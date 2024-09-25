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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import holidays
import os
from sklearn.model_selection import TimeSeriesSplit


####################################################################################################################################
####################################################################################################################################
######################## -----------------------------------------------------------------------------###############################
######################## ---------------- -------------------- DATA ----------------------------------###############################
######################## -----------------------------------------------------------------------------###############################
####################################################################################################################################
####################################################################################################################################


def get_train_data(f_name):
    _target_column_name = "log_bike_count"
    data = pd.read_parquet(f_name)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_final_test_data(path="."):
    f_name = "../input/mdsb-2023/final_test.parquet"
    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort data by date and counter name for correct cross-validation
    data = data.sort_values(["date", "counter_name"])
    return data


def read_data():

    X_train, y_train = get_train_data("../input/mdsb-2023/train.parquet")
    X_test = get_final_test_data()

    return X_train, y_train, X_test


def _merge_external_data_weather(X):
    """
    Merges external weather data into the main DataFrame.
    Input:
    - X: Main DataFrame
    Output:
    - X: Main DataFrame with merged weather data
    """

    file_path = "../input/additional-data/data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext["date"] = pd.to_datetime(df_ext["date"]).astype("datetime64[us]")

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    # Reset index
    X["orig_index"] = np.arange(X.shape[0])

    X_plus_weather = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "ff", "u", "ssfrai", "n", "vv", "rr3", "t"]].sort_values(
            "date"
        ),
        on="date",
    )

    # Clean the merged dataframe
    X_plus_weather["ff"] = X_plus_weather["ff"].fillna(0)
    X_plus_weather["u"] = X_plus_weather["u"].fillna(0)
    X_plus_weather["ssfrai"] = X_plus_weather["ssfrai"].fillna(0)
    X_plus_weather["n"] = X_plus_weather["n"].fillna(0)
    X_plus_weather["vv"] = X_plus_weather["vv"].fillna(0)
    X_plus_weather["rr3"] = X_plus_weather["rr3"].fillna(0)
    X_plus_weather["t"] = X_plus_weather["t"].fillna(0)

    # Sort back to the original order
    X_plus_weather = X_plus_weather.sort_values("orig_index")
    X_plus_weather.drop(columns=["orig_index"], inplace=True)

    return X_plus_weather


def _merge_holidays_week_end(X):
    """
    Adds binary columns for holidays and weekends to the DataFrame.
    Input:
    - X: DataFrame with a 'date' column
    Output:
    - X: DataFrame with 'is_holiday' and 'is_weekend' binary columns
    """

    X = X.copy()

    # Get France holidays using the holidays library
    fr_holidays = holidays.France(years=[2020, 2021])

    # Function to apply to the date column to determine if the date is a holiday
    def is_holiday(date):
        # Normalize the date to remove the time (since holidays are usually considered for the whole day regardless of the time)
        date = date.normalize()
        return int(date in fr_holidays)

    # Function to check if a date is a weekend
    def is_weekend(date):
        return int(date.dayofweek >= 5)  # 5 for Saturday, 6 for Sunday

    # Define the school_holiday_periods with their respective times
    # DATA from : https://vacances-scolaires.education/annee-2020-2021.php
    school_holiday_periods = [
        pd.date_range(start="2020-10-17", end="2020-11-02"),
        pd.date_range(start="2020-12-19", end="2021-01-04"),
        pd.date_range(start="2021-02-13", end="2021-03-01"),
        pd.date_range(start="2021-04-10", end="2021-04-26"),
        pd.date_range(start="2021-05-12", end="2021-05-17"),
        pd.date_range(start="2021-07-06", end="2021-09-01"),
        pd.date_range(start="2021-10-23", end="2021-11-08"),
        pd.date_range(start="2021-12-18", end="2022-01-03"),
    ]

    # Function to check school_holiday status
    def is_school_holiday(date):
        return any(date in period for period in school_holiday_periods)

    # Apply the function to the date column to create a new binary column for holidays
    X["is_holiday"] = X["date"].apply(is_holiday) | X["date"].apply(is_school_holiday)
    X["is_weekend"] = X["date"].apply(is_weekend)
    return X


def _merge_Curfews_lockdowns_COVID(X):
    """
    Adds binary columns for curfew and lockdown status to the DataFrame.
    Input:
    - X: DataFrame with a 'date' column
    Output:
    - X: DataFrame with 'is_lockdown' and 'is_curfew' binary columns
    """

    X = X.copy()

    # Define the lockdown periods
    lockdown_periods = [
        pd.date_range(start="2020-03-17", end="2020-05-11"),
        pd.date_range(start="2020-10-30", end="2020-12-15"),
        pd.date_range(start="2021-04-03", end="2021-05-03"),
    ]

    # Define the curfew periods with their respective times
    curfew_periods = [
        {
            "range": pd.date_range(start="2020-10-17", end="2020-10-29"),
            "start": 21,
            "end": 6,
        },
        {
            "range": pd.date_range(start="2021-01-16", end="2021-03-20"),
            "start": 18,
            "end": 6,
        },
        {
            "range": pd.date_range(start="2021-03-20", end="2021-05-19"),
            "start": 19,
            "end": 6,
        },
        {
            "range": pd.date_range(start="2021-05-19", end="2021-06-09"),
            "start": 21,
            "end": 6,
        },
        {
            "range": pd.date_range(start="2021-06-09", end="2021-06-20"),
            "start": 23,
            "end": 6,
        },
    ]

    # Function to check lockdown status
    def check_lockdown(date):
        return any(date in period for period in lockdown_periods)

    # Function to check curfew status
    def check_curfew(date):
        for period in curfew_periods:
            if date in period["range"]:
                hour = date.hour
                # Check if the time is within the curfew hours
                if (hour >= period["start"]) or (hour < period["end"]):
                    return True
        return False

    # Apply the functions to create new columns
    X["is_lockdown"] = X["date"].apply(check_lockdown)
    X["is_curfew"] = X["date"].apply(check_curfew)

    return X


def _merge_indicators_COVID(X):
    """
    Merges COVID-related indicators into the DataFrame.
    Input:
    - X: Main DataFrame
    Output:
    - X: DataFrame with merged COVID indicators
    """

    file_path = (
        "../input/additional-data/data/table-indicateurs-open-data-dep-2023-COVID.csv"
    )
    columns_COVID = ["date", "lib_dep", "hosp", "rea", "incid_rea", "rad"]
    df_ext = pd.read_csv(file_path, usecols=columns_COVID, parse_dates=["date"])
    df_ext["date"] = pd.to_datetime(df_ext["date"]).astype("datetime64[us]")

    mask_dep = df_ext["lib_dep"] == "Paris"
    df_ext = df_ext[mask_dep]

    df_ext["date"] = pd.to_datetime(df_ext["date"]).astype("datetime64[us]")

    # Define your start and end date as strings
    start_date = str(X["date"].min().strftime("%Y-%m-%d"))
    end_date = str(X["date"].max().strftime("%Y-%m-%d"))

    # Create a mask that selects the rows between the start and end date inclusive
    mask_date = (df_ext["date"] >= start_date) & (df_ext["date"] <= end_date)

    # Use the mask to filter the DataFrame
    df_ext = df_ext.loc[mask_date]

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])

    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "hosp", "rea", "incid_rea", "rad"]].sort_values("date"),
        on="date",
    )

    # Clean the merged dataframe
    X["hosp"] = X["hosp"].fillna(0)
    X["rea"] = X["rea"].fillna(0)
    X["incid_rea"] = X["incid_rea"].fillna(0)
    X["rad"] = X["rad"].fillna(0)

    # Sort back to the original order
    X = X.sort_values("orig_index")
    X.drop(columns=["orig_index"], inplace=True)

    return X


def _merge_road_accidents_by_year(year, start_date, end_date):
    """
    Merges road accident data for a specific year within a given date range.
    Inputs:
    - year: Year for which to merge data.
    - start_date: Start date of the period of interest.
    - end_date: End date of the period of interest.
    Output:
    - df_accidents: DataFrame with merged road accident data for the specified year.
    """
    # Define columns for different data categories
    columns_CARACT = ["Num_Acc", "jour", "mois", "an", "hrmn", "dep", "com", "adr"]
    columns_USAGERS = ["Num_Acc", "grav"]
    columns_VEHICULES = ["Num_Acc", "catv"]

    # Read data for CARACTERISTIQUES, USAGERS, and VEHICULES
    data_CARACT_path = (
        f"../input/additional-data/data/road accident/caracteristiques-{year}.csv"
    )
    data_USAGERS_path = (
        f"../input/additional-data/data/road accident/usagers-{year}.csv"
    )
    data_VEHICULES_path = (
        f"../input/additional-data/data/road accident/vehicules-{year}.csv"
    )
    data_CARACT = pd.read_csv(data_CARACT_path, sep=";", usecols=columns_CARACT)
    data_USAGERS = pd.read_csv(data_USAGERS_path, sep=";", usecols=columns_USAGERS)
    data_VEHICULES = pd.read_csv(
        data_VEHICULES_path, sep=";", usecols=columns_VEHICULES
    )

    # Filter CARACTERISTIQUES data for Paris and within date range
    data_CARACT = data_CARACT[data_CARACT["dep"] == "75"]
    # data_CARACT['date'] = pd.to_datetime(data_CARACT[['an', 'mois', 'jour', 'hrmn']], format='%Y%m%d%H%M').astype('datetime64[us]')
    data_CARACT["date"] = (
        data_CARACT["an"].astype(str)
        + "-"
        + data_CARACT["mois"].astype(str).str.zfill(2)
        + "-"
        + data_CARACT["jour"].astype(str).str.zfill(2)
        + " "
        + data_CARACT["hrmn"].str[:2]
        + data_CARACT["hrmn"].str[2:]
    )
    data_CARACT["date"] = pd.to_datetime(
        data_CARACT["date"], format="%Y-%m-%d %H:%M"
    ).astype("datetime64[us]")
    data_CARACT.drop(["jour", "mois", "an", "hrmn"], axis=1, inplace=True)
    data_CARACT = data_CARACT[
        (data_CARACT["date"] >= start_date) & (data_CARACT["date"] <= end_date)
    ]

    # Merge CARACTERISTIQUES with USAGERS data
    merged_CARACT_USAGERS = pd.merge(
        data_CARACT, data_USAGERS, on="Num_Acc", how="left"
    )
    # Map severity scores and aggregate data
    severity_mapping = {4: 2, 3: 3, 2: 4, 1: 1}
    merged_CARACT_USAGERS["grav"] = merged_CARACT_USAGERS["grav"].map(severity_mapping)

    aggregated_data = (
        merged_CARACT_USAGERS.groupby("Num_Acc")
        .agg({"grav": "max", "Num_Acc": "count"})
        .rename(columns={"Num_Acc": "Count_accidents", "grav": "Max_Grav_accidents"})
    )

    # Reset the index to make 'Num_Acc' a column again
    aggregated_data.reset_index(inplace=True)
    # Merge the aggregated data back with the original DataFrame
    merged_CARACT_USAGERS = pd.merge(
        merged_CARACT_USAGERS.drop("grav", axis=1).drop_duplicates("Num_Acc"),
        aggregated_data,
        on="Num_Acc",
    )

    # Final merge and filtering for VEHICULES
    # Filter for specific vehicle type if needed
    data_VEHICULES = data_VEHICULES[data_VEHICULES["catv"] == 1]
    df_accidents = merged_CARACT_USAGERS[
        merged_CARACT_USAGERS["Num_Acc"].isin(data_VEHICULES["Num_Acc"])
    ]

    return df_accidents


def _merge_road_accidents(X):
    """
    Merges road accident data for all years present in the DataFrame.
    Input:
    - X: Main DataFrame with a 'date' column.
    Output:
    - X: Main DataFrame with merged road accident data.
    """
    # Extract unique years and define date range
    years = X["date"].dt.year.unique()
    start_date = str(X["date"].min().strftime("%Y-%m-%d"))
    end_date = str(X["date"].max().strftime("%Y-%m-%d"))

    # Merge accident data for each year
    df_accidents_list = []
    for year in years:
        df_accidents_year = _merge_road_accidents_by_year(
            str(year), start_date, end_date
        )
        df_accidents_list.append(df_accidents_year)
    df_accidents = pd.concat(df_accidents_list, ignore_index=True)

    # Merge with the main DataFrame using merge_asof
    X = X.copy()
    # Reset index
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_accidents[["date", "Max_Grav_accidents", "Count_accidents"]].sort_values(
            "date"
        ),
        on="date",
    )

    # Clean the merged dataframe
    X["Max_Grav_accidents"] = X["Max_Grav_accidents"].fillna(0)
    X["Count_accidents"] = X["Count_accidents"].fillna(0)

    # Sort back to the original order
    X = X.sort_values("orig_index")
    X.drop(columns=["orig_index"], inplace=True)

    return X


def add_external_data(X):

    X = X.copy()

    # merge original data + external data
    merged_X_train_external_DATA = _merge_external_data_weather(X)

    # merge original data + external data + holidays
    merged_X_train_external_HOLIDAYS = _merge_holidays_week_end(
        merged_X_train_external_DATA
    )

    # merge original data + external data + holidays + data COVID
    merged_X_train_external_HOLIDAYS_COVID = _merge_Curfews_lockdowns_COVID(
        merged_X_train_external_HOLIDAYS
    )
    merged_X_train_external_HOLIDAYS_COVID = _merge_indicators_COVID(
        merged_X_train_external_HOLIDAYS_COVID
    )

    # merge original data + external data + holidays + data COVID + data accidents
    merged_X_train_external_HOLIDAYS_COVID_ACCIDENTS = _merge_road_accidents(
        merged_X_train_external_HOLIDAYS_COVID
    )

    return merged_X_train_external_HOLIDAYS_COVID_ACCIDENTS


####################################################################################################################################
####################################################################################################################################
######################## -----------------------------------------------------------------------------###############################
######################## ---------------------------------- PIPELINE ---------------------------------###############################
######################## -----------------------------------------------------------------------------###############################
####################################################################################################################################
####################################################################################################################################


def get_estimator():
    """
    Creates a pipe which:
        (1) performs all the transformations done by the functions merge_external_data() and _encode_dates(),
        (2) encodes the categorical and numerical data,
        (3) performs a XGBOOST regression with tuned parameters.

    Parameters:
        None

    Returns:
        pipe (sklearn Pipeline objet): the given pipeline
    """

    # Call the merge_external_data function
    merge_external = FunctionTransformer(add_external_data, validate=False)

    # Call the _encode_dates function to split the date column to several columns
    date_encoder = FunctionTransformer(_encode_dates)

    # Encode the final columns
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_id"]
    numeric_encoder = StandardScaler()
    numeric_cols = [
        "t",
        "u",
        "rr3",
        "is_holiday",
        "rea",
        "rad",
        "hosp",
        "year",
        "month",
        "day",
        "weekday",
        "hour",
    ]

    # Create a ColumnTransformer object to perform all encodings
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_encoder, categorical_cols),
            ("numeric", numeric_encoder, numeric_cols),
        ]
    )

    params = {
        "objective": "reg:squarederror",
        "base_score": None,
        "booster": None,
        "callbacks": None,
        "colsample_bylevel": None,
        "colsample_bynode": None,
        "colsample_bytree": None,
        "device": None,
        "early_stopping_rounds": None,
        "enable_categorical": False,
        "eval_metric": None,
        "feature_types": None,
        "gamma": 0,
        "grow_policy": None,
        "importance_type": None,
        "interaction_constraints": None,
        "learning_rate": 0.02,
        "max_bin": None,
        "max_cat_threshold": None,
        "max_cat_to_onehot": None,
        "max_delta_step": None,
        "max_depth": 13,
        "max_leaves": None,
        "min_child_weight": 5,
        "monotone_constraints": None,
        "multi_strategy": None,
        "n_estimators": 1500,
        "n_jobs": None,
        "num_parallel_tree": None,
        "random_state": None,
        "reg_alpha": None,
        "reg_lambda": None,
        "sampling_method": None,
        "scale_pos_weight": None,
        "subsample": 0.7,
        "tree_method": None,
        "validate_parameters": None,
        "verbosity": None,
        "verbose": True,
        "early_stopping": True,
    }

    # Create the regressor object
    regressor = xgb.XGBRegressor(**params)

    # Create pipeline
    pipe = make_pipeline(merge_external, date_encoder, preprocessor, regressor)

    return pipe


def submission_kaggle(model, X_final_test):
    y_pred = model.predict(X_final_test)
    print(y_pred)
    results = pd.DataFrame(
        dict(
            Id=X_final_test.index,
            log_bike_count=y_pred,
        )
    )
    results.to_csv("submission.csv", index=False)


####################################################################################################################################
####################################################################################################################################
######################## -----------------------------------------------------------------------------###############################
######################## ---------------- -------------------- MAIN ----------------------------------###############################
######################## -----------------------------------------------------------------------------###############################
####################################################################################################################################
####################################################################################################################################


def main():

    # Read data for training
    X_train, y_train, X_test = read_data()

    # Create pipeline
    pipe = get_estimator()

    # Train pipe
    pipe.fit(X_train, y_train)

    # Predict and create csv submission
    submission_kaggle(pipe, X_test)


if __name__ == "__main__":
    main()
