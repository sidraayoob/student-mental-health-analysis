# preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Custom cleaning logic here (fill missing, drop unused, etc.)
    Update this with your actual cleaning logic from the notebook.
    """
    df = df.dropna()  # Example
    return df

def preprocess_data(df, target=None):
    df_copy = df.copy()
    if target and target in df_copy.columns:
        X = df_copy.drop(target, axis=1)
        y = df_copy[target]
        return X, y
    return df_copy, None

def load_and_preprocess(csv_path, target="Academic Performance Change"):
    df = pd.read_csv(csv_path)
    cleaned_df = clean_data(df)
    X, y = preprocess_data(cleaned_df, target=target)
    return X, y

if __name__ == '__main__':
    X, y = load_and_preprocess(
        'C:\\Users\\HelpTech\\OneDrive\\Desktop\\End-to-End Machine Learning Project\\data\\Student Mental Health Analysis During Online Learning.csv',
        target="Academic Performance Change"
    )
    print("Data preprocessing complete. Features shape:", X.shape, "Target shape:", y.shape)
