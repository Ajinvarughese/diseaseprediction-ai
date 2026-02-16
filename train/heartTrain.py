# ==============================
# Importing Dependencies
# ==============================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ==============================
# Load Dataset
# ==============================

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "dataset", "heart.csv")

heart_data = pd.read_csv(file_path)

print("Class Distribution:")
print(heart_data['target'].value_counts())

# ==============================
# Split Features & Target
# ==============================

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# ==============================
# Train Test Split
# ==============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# ==============================
# Model (Scaler + Logistic Regression)
# ==============================

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    ))
])

model.fit(X_train, Y_train)

# ==============================
# Evaluation
# ==============================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Test Accuracy:", accuracy_score(Y_test, test_pred))

# ==============================
# Test Prediction Example
# ==============================

input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

input_df = pd.DataFrame([input_data], columns=X.columns)

proba = model.predict_proba(input_df)[0][1]

print("Probability of disease:", proba)

if proba > 0.65:
    print("Has Heart Disease")
else:
    print("No Heart Disease")

# ==============================
# Save Model & Columns
# ==============================

model_dir = os.path.join(current_dir, "..", "saved_models")
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_dir, "heart_disease_model.sav")
pickle.dump(model, open(model_path, 'wb'))

# Save column order
columns_path = os.path.join(model_dir, "heart_columns.pkl")
pickle.dump(X.columns.tolist(), open(columns_path, "wb"))

print("Model and columns saved successfully.")
