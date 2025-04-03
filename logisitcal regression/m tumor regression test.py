import numpy as np
import pandas as pd

import pickle
from sklearn.preprocessing import StandardScaler


dfTrain = pd.read_csv('./logisitcal regression/breast-cancer-tumor-data.csv')
dfTest = pd.read_csv('./logisitcal regression/breast-cancer-tumor-test-data.csv')

df = pd.concat([dfTrain, dfTest])
print(df)

df.drop(['id'], axis=1, inplace=True) #remove id column cause thats useless

df['diagnosis'].replace({'M': 'malignant', 'B': 'benign'}, inplace=True)
dfTest['diagnosis'].replace({'M': 'malignant', 'B': 'benign'}, inplace=True) #for test data when we read it later

df_correlations = pd.get_dummies(df, columns=["diagnosis"], dtype=int) #make diagnosis column target column
correlation = df_correlations.corr()
corr_benign = correlation["diagnosis_benign"].drop(["diagnosis_benign", "diagnosis_malignant"]).abs()
highest_corr_benign = corr_benign.idxmax()
highest_value_benign = corr_benign.max()

corr_malignant = correlation["diagnosis_malignant"].drop(["diagnosis_benign", "diagnosis_malignant"]).abs()
highest_corr_malignant = corr_malignant.idxmax()
highest_value_malignant = corr_malignant.max() 
x = df.drop("diagnosis", axis=1, inplace=False).values
y = df['diagnosis'].values
scaler = StandardScaler()

scaler.fit_transform(x)
x_test = scaler.transform(x)
with open("./logisitcal regression/m-tumor_regression_model.pkl", "rb") as f:
    model = pickle.load(f)


y_pred = model.predict(x_test)


print(f"Model Classification: {y_pred[-1]}")
print(f"Actual classification: {dfTest['diagnosis'].values[-1]}")