#Importing Libraries

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning Libraries
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

#Importing Function from Evaluation.py
from Evaluation import elbow_method, evaluate_clustering

#loading the csv created from feature engineering
df = pd.read_csv("rfm.csv")

#function for hierarchial clustering
def hierarchical_clustering(data, k):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    return labels

#Scale Features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Calling Hierarchial Function 
labels = hierarchical_clustering(scaled_data, 4)
df["Cluster"] = labels

#Performing Evaluations
elbow_method(scaled_data)
results = evaluate_clustering(scaled_data, labels)
print(results)
