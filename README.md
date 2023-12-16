# Bike counters Paris Project

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

## Getting started

### Download the data,

Download the data files,
 - [train.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet)
 - [test.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/test.parquet)

and put them into into the data folder.

Note that the `test.parquet` file is different from the actual `final_test.parquet` used for the evaluation on Kaggle. This file is just here for convenience.

### Install

It is recommended to create a new virtual environement for this project. For instance, with conda,
```bash
conda create -n bikes-count python=3.10
y
```

You can install the dependencies with the following command-line:

```bash
pip install -U -r requirements.txt
```

## Project organization

### data/
This folder contains various datasets used for training and evaluating our model. The datasets are organized into separate files, each representing different aspects or sources of data.
- `train.parquet` : the original train data given
- `test.parquet` : the original test data given
- `external_data.csv` : the weather data given
- ...

### utils/
This directory includes utility scripts that support data operations:
- `get_data.py`: Functions for loading, cleaning, and merging datasets to prepare them for analysis and training.

### submissions/
Contains Python function modules that are used to format and submit predictions to Kaggle competitions. These functions ensure that submissions adhere to the competition's requirements.
- `estimator_submission.py`: Python file for submission of our final solution on Kaggle.

### Notebooks
Several Jupyter notebooks are available for elaborating strategies, exploratory data analysis, model experimentation, and final model execution:
- `Metadata_stategy.ipynb`: For presenting metadata from each datasets used and collaborate on strategies
- `Data_exploration.ipynb`: For initial exploration and visualization of the datasets and external datasets.
- `model_tuning.ipynb`: Used to fine-tune the model's hyperparameters.
- `Training_model.ipynb`: For merging all data, preprocess it, testing it on different models and visualize final results.
- `Training_model_pipeline_final.ipynb`: Pipeline of the best model obtained from `Training_model.ipynb` with predictions of final test dataset.

### Submissions

We worked first with python file for submission on kaggle then switched to local creation of the csv. We forgot to switch back to the original method preferred by the challenge. The file `estimator_submission.py` contains what should have been put in the kaggle challenge.

## Authors

**Alexandre Brun** and **Thomas Bordes** using the initial git by teachers of the course **Python for Data Science** from the master Data Science for Business **X - HEC**.

