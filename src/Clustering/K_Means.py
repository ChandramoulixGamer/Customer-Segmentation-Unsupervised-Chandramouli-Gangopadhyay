#Importing Libraries

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Importing Function from Evaluation.py
from Evaluation import elbow_method, evaluate_clustering

#loading the csv created from feature engineering
df = pd.read_csv("rfm.csv")

#Function for K_means clustering
def kmeans_clustering(data, k):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)
    return labels, model

#Scale Features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Calling K_means Function 
k = 4
labels, model = kmeans_clustering(scaled_data, k)
df["Cluster"] = labels

#Performing Evaluations
elbow_method(scaled_data)
results = evaluate_clustering(scaled_data, labels)
print(results)




