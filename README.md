# 🚴‍♂️ Bike Counters Paris Project

![GH Actions](https://github.com/ramp-kits/bike_counters/actions/workflows/main.yml/badge.svg)

Welcome to the **Bike Counters Paris Project**! This project focuses on analyzing bicycle traffic in Paris, leveraging both historical bike counter data and external data sources like weather conditions. Below, you'll find everything you need to get started with the project, including setup instructions, data details, and project organization.

---

## 🚀 Getting Started

### 📥 Download the Data

The data was downloaded from the links provided in [Bikes_data_strategy.pdf](Bikes_data_strategy.pdf).

### 🛠️ Installation

We recommend creating a new virtual environment for this project. If you are using Conda, run the following:

```bash
conda env create -f environment.yml
```

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

### Submissions
Initially, submissions were made using Python scripts, but we later switched to CSV creation for local testing. The `estimator_submission.py` file contains the original method for Kaggle submission, as required by the challenge.


## 📸 Screenshots

Main Interface:

## 💻 Tech Stack

- **Python**: Programming language
- **sklearn - XGBoost**: Deep learning models
- **Streamlit**: Web framework for building interactive web applications

## 👥 Authors

- **Alexandre Brun**
- **Thomas Bordes**

This project was built using the initial GitHub repository provided by instructors of the **Python for Data Science** course from the **X - HEC Master’s Program** in Data Science for Business.

## 📚 Acknowledgements

> 🔍 **Code inspired by**: [SkalskiP/yolov8-live](https://github.com/SkalskiP/yolov8-live/tree/master)  
> 🎨 **Streamlit UI inspired by**: [tyler-simons/BackgroundRemoval](https://github.com/tyler-simons/BackgroundRemoval/tree/main)
