#Importing silhouette score and davies bouldin score
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#Loading rfm_scaled
rfm_scaled = pd.read_csv("rfm_scaled.csv")

#K-Means Clustering and Initialization

def kmeans_clustering(data, k):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)
    return labels, model
labels_kmeans, model = kmeans_clustering(rfm_scaled, k=4)

#Evaluation

#Elbow Method Function
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

#Cluster Evaluation
def evaluate_clustering(data, labels):

    sil_score = silhouette_score(data, labels)
    db_score = davies_bouldin_score(data, labels)

    return {
        "Silhouette Score": sil_score,
        "Davies Bouldin Score": db_score
    }

#Cluster_Summary
def cluster_summary(rfm_scaled, labels):
    rfm_scaled['Cluster'] = labels
    summary = rfm_scaled.groupby('Cluster').mean()
    return summary
summary = cluster_summary(rfm_scaled, labels_kmeans)
print(summary)

#Cluster Interpretation
def interpret_clusters(summary):

    print("\n===== CLUSTER INTERPRETATION =====\n")

    interpretations = []

    for cluster in summary.index:

        rec = summary.loc[cluster, 'Recency']
        freq = summary.loc[cluster, 'Frequency']
        mon = summary.loc[cluster, 'Monetary']

        # Default 
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

summary = cluster_summary(rfm_scaled, labels_kmeans)

interpretations = interpret_clusters(summary)
