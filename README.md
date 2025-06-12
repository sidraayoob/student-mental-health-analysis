# Student Mental Health Analysis During Online Learning

This project is an end-to-end machine learning application that predicts changes in students' academic performance based on their mental health and lifestyle factors during online learning. The app provides a user-friendly interface for making predictions and includes all steps from data preprocessing to model deployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)


## Overview

The goal of this project is to analyze and predict how various factors such as screen time, sleep duration, physical activity, stress, and anxiety affect students' academic performance during online learning. The solution uses a Random Forest Classifier and provides a Streamlit-based web interface for predictions.

## Features

- Data cleaning and preprocessing
- Exploratory data analysis (EDA) in Jupyter Notebook
- Random Forest model training and evaluation
- Model serialization with pickle
- Streamlit GUI for user-friendly predictions

## Dataset

The dataset is included as `Student Mental Health Analysis During Online Learning.csv`. It contains the following columns:

- Name
- Gender
- Age
- Education Level
- Screen Time (hrs/day)
- Sleep Duration (hrs)
- Physical Activity (hrs/week)
- Stress Level
- Anxious Before Exams
- Academic Performance Change (target)

## Requirements

- Python 3.8 or higher
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pickle

Install the required packages using the provided `requirements.txt` file.

## Installation

1. **Clone the repository** (if applicable) or download the project files.
2. **Install dependencies** using pip:

   ```sh
   pip install -r requirements.txt
   ```

## How to Run

1. Run `preprocess.py`

   ```sh
   cd src
   python preprocess.py
   ```
2. Run `train_model.py` to train and save the model.

   ```sh
   cd src
   python train_model.py
   ```
3. Run `gui_app.py` with Streamlit:

   ```sh
   cd src
   streamlit run gui_app.py
   ```

## Project Structure

```
.
├── data
│   └── Student Mental Health Analysis During Online Learning.csv
├── notebooks
│   └── EDA_and_Model_Training.ipynb
├── src
│   ├── train_model.py
│   ├── gui_app.py
│   └── preprocess.py
├── requirements.txt
└── README.md
```

