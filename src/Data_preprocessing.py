#Importing Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

#Function to load the dataset

def load_data(path):
    df = pd.read_csv(path)
    return df

#Functions to remove missing data

def handle_missing_values(df):
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna("Unknown")
    return df


def remove_outliers(df, cols):

    for col in cols:

     Q1 = df[['Quantity','UnitPrice']].quantile(0.25)
     Q3 = df[['Quantity','UnitPrice']].quantile(0.75)

     IQR = Q3 - Q1

    df = df[~((df[['Quantity','UnitPrice']] < (Q1 - 1.5 * IQR)) |
        (df[['Quantity','UnitPrice']] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df

#function call and print

df= load_data("Raw_Dataset.Csv")
df.to_csv("Cleaned_Dataset.Csv",index=False)
df = handle_missing_values(df)
numeric_cols = ['Quantity', 'UnitPrice']
df = remove_outliers(df, numeric_cols)
print(df)