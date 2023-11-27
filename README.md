# Starting kit on the bike counters dataset

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

## Getting started

### Download the data,

Download the data files,
 - [train.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet)
 - [test.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/test.parquet)

and put them into into the data folder.

Note that the `test.parquet` file is different from the actual `final_test.parquet` used for the evaluation on Kaggle. This file is just here for convenience.

### Install

To run the notebook you will need the dependencies listed
in `requirements.txt`. 

It is recommended to create a new virtual environement for this project. For instance, with conda,
```bash
conda create -n bikes-count python=3.10
y
```

You can install the dependencies with the following command-line:

```bash
pip install -U -r requirements.txt
```


### `data/`
This folder contains various datasets used for training and evaluating our model. The datasets are organized into separate files, each representing different aspects or sources of data.
- `train.parquet` : the original train data given
- `test.parquet` : the original test data given
- `external_data.csv` : the weather data given
- ...

### Notebooks
Several Jupyter notebooks are available for elaborating strategies, exploratory data analysis, model experimentation, and final model execution:
- `bike_counters_starting_kit.ipynb` : Initial file given to explore data
- `Metadata_stategy.ipynb`: For presenting metadata from each datasets used and collaborate on strategies
- `Data_exploration.ipynb`: For initial exploration and visualization of the datasets.
- `model_tuning.ipynb`: Used to fine-tune the model's hyperparameters.
- `Training_model.ipynb`: Executes the final model training and prediction pipeline.

### `submissions/`
Contains Python function modules that are used to format and submit predictions to Kaggle competitions. These functions ensure that submissions adhere to the competition's requirements.

### `utils/`
This directory includes utility scripts that support data operations:
- `get_data.py`: Functions for loading, cleaning, and merging datasets to prepare them for analysis.
- `training_utilities.py`: Functions dedicated to configuring, training, and evaluating the machine learning model.




### Submissions

Upload your script file `.py` to Kaggle using the Kaggle interface directly.
The platform will then execute your code to generate your submission csv file, and compute your score.
Note that your submission .csv file must have the columns "Id" and "bike_log_count", and be of the same length as `final_test.parquet`.

## Authors

**Alexandre Brun** and **Thomas Bordes** using the initial git by teachers of the course **Python for Data Science** from the master Data Science for Business **X - HEC**.

