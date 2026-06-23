"""
Retrain and save model using this environment's sklearn version.
Run with: python retrain.py
"""
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.path.join('..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(DATA_PATH)

# --- Cleaning ---
df.drop(columns=['customerID'], inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop_duplicates(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
               'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0,
                           'No phone service': 0, 'No internet service': 0})

df = pd.get_dummies(df, columns=['gender', 'InternetService', 'Contract', 'PaymentMethod'],
                    drop_first=False)

# --- Feature engineering ---
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['HasMultipleServices'] = (
    df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies']].sum(axis=1) >= 3
).astype(int)

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# --- Train ---
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --- Save ---
os.makedirs('../models', exist_ok=True)
joblib.dump(model,                 '../models/churn_model.pkl')
joblib.dump(scaler,                '../models/scaler.pkl')
joblib.dump(list(X_train.columns), '../models/feature_columns.pkl')

print(f"Saved model with sklearn {__import__('sklearn').__version__}")
print(f"Features: {len(X_train.columns)}")
