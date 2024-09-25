# 🚴‍♂️ Bike Counters Paris Project

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

Welcome to the **Paris Bike Count Analysis** project! This repository analyzes bicycle traffic 🚴‍♀️🚴‍♂️ in Paris using historical data and external sources (weather ☁️, accidents 🚧, COVID-19 🦠) to predict bike counts across different locations in the city.

## 📁 Project structure

```bash
.
├── data
        ├── train.parquet                                    # Original training data
        ├── test.parquet                                     # Original test data
        ├── external_data.csv                                # Weather data
        ├── road accident                                    # Road accident data
            ├── ...
        ├── table-indicateurs-open-data-dep-2023-COVID.csv   # COVID data
├── deliverables                                              
        ├── Alexandre_Brun_Thomas_Bordes_bike_count_2023.pdf # Final repport
        ├── Bikes_data_strategy.pdf                          # What data are used and why
        ├── Presentation.pptx                                # Power Point Final presentation
├── submissions                                              # Code to push results on Kaggle
        ├── ...       
├── utils                
        ├── get_data.py                                      # Python code for merging external data 
├── Bikes_data_strategy.pdf                                  # Overview of the metadata for each dataset and initial strategy discussions.

├── Training_model.ipynb                                     # End-to-end model training, data merging, preprocessing, and result visualization.
├── Training_model_pipeline_final.ipynb                      # Pipeline for the best-performing model.
```

## ✨ Features

- 🍃 Data Merging: Combine multiple external datasets (weather ☀️, accidents 🚧, etc.) to enhance bike count predictions.
- 🧠 Model Training: Train machine learning models using scikit-learn, XGBoost, and others to predict bike counts 🚴‍♂️.
- 📊 Prediction Visualization: Generate detailed visualizations 📈 for bike count predictions.

## 🚀 Quick Start

Follow these instructions to get the project up and running on your local machine.

### 1. Clone the repository:

```bash
git clone https://github.com/ThomasBrdes/bike-count-DSB.git
cd bike-count-DSB
```

### 2. Install env:

```bash
conda env create -f environment.yml
conda activate bikes-count
```

### 3. Launch Jupyter:

```bash
jupyter notebook
```

## 📥 Download the Data

The data was downloaded from the links provided in [Bikes_data_strategy.pdf](Bikes_data_strategy.pdf).


## Submissions
Initially, submissions were made using Python scripts, but we later switched to CSV creation for local testing. The `estimator_submission.py` file contains the original method for Kaggle submission, as required by the challenge.


## 📸 Screenshots

Main Interface:

## 🛠️ Tech Stack

- **Python** 🐍: Programming language
- **scikit-learn - XGBoost** 🤖: For training and predictions.
- **Streamlit** 🖥️: Web framework for building interactive web applications

## 👥Authors

 - **Alexandre Brun** 🧑‍💻
**Thomas Bordes** 🧑‍💻

This project is part of the Python for Data Science course from the **X-HEC Master’s Program** 🏫.

## 📚 Acknowledgements

> 🔍 **Code inspired by**: [SkalskiP/yolov8-live](https://github.com/SkalskiP/yolov8-live/tree/master)  
> 🎨 **Streamlit UI inspired by**: [tyler-simons/BackgroundRemoval](https://github.com/tyler-simons/BackgroundRemoval/tree/main)
