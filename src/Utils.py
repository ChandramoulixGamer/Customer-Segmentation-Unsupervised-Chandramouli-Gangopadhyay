#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt

#Function to save modified data to csv
def save_csv(df, filename):
    df.to_csv(filename, index=False)

#Function for cluster evaluation
def cluster_distribution(df):
    return df['Cluster'].value_counts()

#Function for result based on evaluation
def cluster_summary(df):
    return df.groupby("Cluster").mean()