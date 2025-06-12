# src/train_model.py

import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess

def train_and_save_model():
    # Correct path to your dataset
    data_path = os.path.join('..', 'data', 'C:\\Users\\HelpTech\\OneDrive\\Desktop\\End-to-End Machine Learning Project\\data\\Student Mental Health Analysis During Online Learning.csv')
    X, y = load_and_preprocess(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join('..', 'data', 'student_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved at:", model_path)

if __name__ == '__main__':
    train_and_save_model()
