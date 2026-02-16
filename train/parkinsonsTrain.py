# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "dataset", "parkinsons.csv")

# Data Collection & Analysis
parkinsons_data = pd.read_csv(file_path)

parkinsons_data.head()

parkinsons_data.shape

parkinsons_data.info()

parkinsons_data.isnull().sum()

parkinsons_data.describe()

parkinsons_data['status'].value_counts()

parkinsons_data.groupby('status').mean(numeric_only=True)

# Data Pre-Processing
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

print(X)
print(Y)

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# Model Training
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

# Building a Predictive System
input_data = (
    197.07600,206.89600,192.05500,0.00289,0.00001,
    0.00166,0.00168,0.00498,0.01098,0.09700,
    0.00563,0.00680,0.00802,0.01689,0.00339,
    26.77500,0.422229,0.741367,-7.348300,
    0.177551,1.743867,0.085569
)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The Person does not have Parkinsons Disease")
else:
    print("The Person has Parkinsons")


# Create path relative to this script
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "..", "saved_models")

# Ensure folder exists
os.makedirs(model_dir, exist_ok=True)

# Full model path
model_path = os.path.join(model_dir, "parkinsons_model.sav")

# Save model
pickle.dump(model, open(model_path, 'wb'))

# Load model
loaded_model = pickle.load(open(model_path, 'rb'))


for column in X.columns:
    print(column)
