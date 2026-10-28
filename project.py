# Diabetes Prediction Model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
dataset = pd.read_csv("diabetes.csv")

# Separate Features and Target
X = dataset.drop(columns='Outcome', axis=1)
Y = dataset['Outcome']

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, Y_train)

# Model Evaluation
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy Score =", test_accuracy)

# -----------------------------------
# User Input for Prediction
# -----------------------------------

print("\nEnter the following values:")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose Level: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin Level: "))
bmi = float(input("BMI: "))
diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]])

# Standardize Input Data
std_data = scaler.transform(input_data)

# Prediction
prediction = model.predict(std_data)

if prediction[0] == 0:
    print("\n✅ The person is NOT diabetic")
else:
    print("\n⚠️ The person IS diabetic")