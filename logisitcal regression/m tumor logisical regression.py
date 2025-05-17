import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report



df = pd.read_csv('./logisitcal regression/breast-cancer-tumor-data.csv')
df.drop(['id'], axis=1, inplace=True) #remove id column cause thats useless

df['diagnosis'].replace({'M': 'malignant', 'B': 'benign'}, inplace=True) #TEMP

df_correlations = pd.get_dummies(df, columns=["diagnosis"], dtype=int) #make diagnosis column target column


correlation = df_correlations.corr()

# get highest correlation value to diagnosis, this val is concave points_worst for both
corr_benign = correlation["diagnosis_benign"].drop(["diagnosis_benign", "diagnosis_malignant"]).abs()
highest_corr_benign = corr_benign.idxmax()
highest_value_benign = corr_benign.max()


corr_malignant = correlation["diagnosis_malignant"].drop(["diagnosis_benign", "diagnosis_malignant"]).abs()
highest_corr_malignant = corr_malignant.idxmax()
highest_value_malignant = corr_malignant.max() 


print(df.head())

#create train data
x = df.drop("diagnosis", axis=1, inplace=False).values
y = df['diagnosis'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#actual model
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


with open("./logisitcal regression/m-tumor_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)
