# Importing Libraries
# Data Handling
import pandas as pd
import numpy as np
import datetime as dt
import os

# Data Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning (Sklearn)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Function to load the dataset

def load_data(path):
    df = pd.read_csv(path)
    return df

# Functions to remove missing data

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

# Feature Engineering

# Behavioural Features
def behavioural_features(df):

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': 'sum'
    })
    #Renaming Columns 
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Revenue': 'Monetary'
    }, inplace=True)

    # Debug check
    print("Columns after rename:", rfm.columns)

    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Purchase_Rate'] = rfm['Frequency'] / (rfm['Recency'] + 1)

    return rfm

# Scale Features
def scale_features(rfm):
    scaler = StandardScaler()
    
    rfm_numeric = rfm.select_dtypes(include=['int64', 'float64'])
    
    scaled_data = scaler.fit_transform(rfm_numeric)
    
    rfm_scaled = pd.DataFrame(
        scaled_data,
        columns=rfm_numeric.columns,
        index=rfm.index
    )
    
    return rfm_scaled
# Derived Ratios
def add_derived_ratios(rfm):

    rfm = rfm.copy()

    # Avoid division by zero
    rfm['Avg_Order_Value'] = rfm['Monetary'] / (rfm['Frequency'] + 1e-6)
    rfm['Purchase_Rate'] = rfm['Frequency'] / (rfm['Recency'] + 1e-6)

    # Additional advanced ratios
    rfm['Value_per_Day'] = rfm['Monetary'] / (rfm['Recency'] + 1e-6)
    rfm['Engagement_Score'] = rfm['Frequency'] * rfm['Monetary']

    return rfm


# Clustering

# K-Means Clustering
def kmeans_clustering(data, k):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)
    return labels, model

# Hierarchial Clustering
def hierarchical_clustering(data, k):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    return labels

# DB Scan
def dbscan_clustering(data, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels

#Gaussian Mixture Model
def gmm_clustering(data, k):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(data)
    return labels

# Evaluation

# Elbow Method 
def elbow_method(data):
    
    inertia = []

    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        inertia.append(model.inertia_)

    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

# Cluster Evaluation
def evaluate_clustering(data, labels):

    sil_score = silhouette_score(data, labels)
    db_score = davies_bouldin_score(data, labels)

    return {
        "Silhouette Score": sil_score,
        "Davies Bouldin Score": db_score
    }

#Visualization.py

def pca_visualization(data, labels):

    os.makedirs("results/cluster_plots", exist_ok=True)

    # 2D PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Cluster Visualization (2D)")
    plt.savefig("results/cluster_plots/pca_2d.png")
    plt.show()

    # 3D PCA 
    pca_3d = PCA(n_components=3)
    reduced_3d = pca_3d.fit_transform(data)

   

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], c=labels)

    ax.set_title("PCA Cluster Visualization (3D)")
    plt.savefig("results/cluster_plots/pca_3d.png")
    plt.show()

#Tsne Visualization
def tsne_visualization(data, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(data)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Cluster Visualization (2D)")
    plt.savefig("results/cluster_plots/tsne_2d.png")
    plt.show()

# Business_insights.py

def cluster_summary(df, labels):
    df = df.copy()   
    df['Cluster'] = labels
    summary = df.groupby('Cluster').mean()
    return summary

def interpret_clusters(summary):

    print("\n===== CLUSTER INTERPRETATION =====\n")

    interpretations = []

    for cluster in summary.index:

        rec = summary.loc[cluster, 'Recency']
        freq = summary.loc[cluster, 'Frequency']
        mon = summary.loc[cluster, 'Monetary']

        # Default values
        customer_type = ""
        behavior = ""
        spending = ""
        demographics = ""

        # Logic for interpretation
        if mon > summary['Monetary'].mean() and freq > summary['Frequency'].mean():
            customer_type = "High-value premium buyers"
            behavior = "Frequent purchases, highly engaged"
            spending = "High spending per transaction"
            demographics = "Likely loyal, long-term customers"

        elif freq > summary['Frequency'].mean() and mon < summary['Monetary'].mean():
            customer_type = "High-frequency low spenders"
            behavior = "Frequent but small purchases"
            spending = "Low to moderate spending"
            demographics = "Price-sensitive customers"

        elif rec > summary['Recency'].mean():
            customer_type = "At-risk customers"
            behavior = "Inactive for a long time"
            spending = "Declining spending trend"
            demographics = "Likely to churn"

        else:
            customer_type = "Budget occasional shoppers"
            behavior = "Infrequent purchases"
            spending = "Low spending"
            demographics = "Casual or new customers"

        print(f"\n🔹 Cluster {cluster}")
        print(f"Type: {customer_type}")
        print(f"Behavior: {behavior}")
        print(f"Spending: {spending}")
        print(f"Demographics: {demographics}")
        print("-" * 40)

        interpretations.append({
            "Cluster": cluster,
            "Customer_Type": customer_type,
            "Behavior": behavior,
            "Spending": spending,
            "Demographics": demographics
        })

    return interpretations

def generate_business_insights(rfm, labels):
    df = rfm.copy()
    df['Cluster'] = labels

    # Aggregate cluster data
    summary = df.groupby('Cluster').agg({
        'Monetary': 'mean',
        'Frequency': 'mean',
        'Recency': 'mean'
    })

    print("\n===== BUSINESS INSIGHTS =====\n")

    insights = []

    # Identify key segments
    highest_revenue_cluster = summary['Monetary'].idxmax()
    premium_cluster = summary['Frequency'].idxmax()
    churn_cluster = summary['Recency'].idxmax()
    retention_cluster = summary['Recency'].sort_values(ascending=False).index[1]

    for cluster in summary.index:

        revenue = summary.loc[cluster, 'Monetary']
        freq = summary.loc[cluster, 'Frequency']
        rec = summary.loc[cluster, 'Recency']

        print(f"\n🔹 Cluster {cluster}")
        print(f"Avg Revenue: {revenue:.2f}, Frequency: {freq:.2f}, Recency: {rec:.2f}")

        # Strategy logic
        if cluster == highest_revenue_cluster:
            strategy = "Focus on retention and loyalty programs"
            offer = "Exclusive VIP discounts, early access to products"
            tag = "Highest Revenue Segment"

        elif cluster == premium_cluster:
            strategy = "Upsell premium products and memberships"
            offer = "Bundle offers, subscription plans"
            tag = "Premium Marketing Segment"

        elif cluster == churn_cluster:
            strategy = "Re-engagement campaigns needed"
            offer = "Heavy discounts, win-back campaigns"
            tag = "High Churn Risk"

        elif cluster == retention_cluster:
            strategy = "Retention strategies required"
            offer = "Personalized offers, reminders"
            tag = "Needs Retention"

        else:
            strategy = "Maintain engagement"
            offer = "Regular promotions"
            tag = "General Segment"

        print(f"Tag: {tag}")
        print(f"Strategy: {strategy}")
        print(f"Offer: {offer}")

        insights.append({
            "Cluster": cluster,
            "Tag": tag,
            "Strategy": strategy,
            "Offer": offer
        })

    insights_df = pd.DataFrame(insights)

    return summary, insights_df


#Load Data
df= load_data("Raw_Dataset.Csv")

#Storing the result new data to save the original dataset from any update
df.to_csv("Cleaned_Dataset.Csv",index=False)

#Preprocessing Function call
df = handle_missing_values(df)
numeric_cols = ['Quantity', 'UnitPrice']
df = remove_outliers(df, numeric_cols)

#printing the cleaned data
print(df)

#Loading the Cleaned Dataset
df = load_data("Cleaned_Dataset.csv")

#Applying Feature Engineering
#Function Call
rfm_behavioural_features = behavioural_features(df)
rfm_scaled_features = scale_features(rfm_behavioural_features)
rfm = add_derived_ratios(rfm_behavioural_features)

#Saving The New Datasets
rfm_behavioural_features.to_csv("rfm.csv",index=False)
rfm_scaled_features.to_csv("rfm_scaled.csv",index=False)

#Loading rfm dataset
df = pd.read_csv("rfm.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Model Selection
elbow_method(scaled_data)
labels_kmeans, model = kmeans_clustering(scaled_data, k=4)
silhouette, db_index = evaluate_clustering(scaled_data, labels_kmeans)

#Printing Scores
print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", db_index)

# Business Insights
summary = cluster_summary(df, labels_kmeans)
print(summary)
summary = cluster_summary(rfm, labels_kmeans)

interpretations = interpret_clusters(summary)
summary, insights = generate_business_insights(rfm_behavioural_features, labels_kmeans)

print("\nCluster Summary:\n", summary)

insights.to_csv("results/metrics/business_insights.csv", index=False)

summary.to_csv("Cluster_Summary.csv",index=False)
print("\n=== FINAL INSIGHTS TABLE ===")
print(insights)
