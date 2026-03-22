Capstone_Project
# AI-Driven Customer Intelligence System for Strategic Business Decision Making
1. Problem Statement

Businesses often struggle to understand customer behavior and segment their audience effectively.  
This project aims to perform **customer segmentation** using clustering techniques based on purchasing behavior, helping businesses:

Identify high-value customers  
Detect churn risk  
Improve marketing strategies  



2. Dataset Description

The dataset contains transactional data of an online retail store, including:

InvoiceNo → Unique transaction ID  
StockCode → Product ID  
Description → Product description  
Quantity → Number of items purchased  
InvoiceDate → Transaction date  
UnitPrice → Price per item  
CustomerID → Unique customer ID  
Country → Customer location  

From this, **RFM features** were created:

**Recency (R)** → Days since last purchase  
**Frequency (F)** → Number of purchases  
**Monetary (M)** → Total spending  

Additional behavioral features:
Avg Order Value  
Purchase Rate  


3. Algorithms Used

The following clustering algorithms were implemented and compared:

**K-Means Clustering**
**DBSCAN (Density-Based Clustering)**
**Hierarchical (Agglomerative) Clustering**
**Gaussian Mixture Model**


4. How to Run the Project

# Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

# Step 2: Run the Project
```bash
python main.py
```

5. Key Results

# Optimal Number of Clusters: 4
# Best Algorithm: K-Means (based on Silhouette Score & Davies-Bouldin Score)
# Business Insights:
# High Revenue Segment: Customers with high Monetary + Frequency
# Premium Marketing Target: Loyal and high-spending customers
# Churn Risk Group: High Recency, low Frequency customers
# Retention Needed: Medium spenders with declining activity

6. Sample Visualizations

The project includes the following visual outputs:

# Elbow Method Graph
# Cluster Distribution Plots
# PCA-based Customer Segmentation Visualization
# Correlation Heatmap

All Stored in Results Folder
# Example:
![Elbow Method Graph](image.png)
![Feature Collrelation Heatmap](image-1.png)

7. Project Structure
# Data
# Notebooks
# Reports
# Results
# SRC.py
# Main.py
# Requirements.txt

8. Conclusion

This project successfully segments customers using clustering techniques, enabling:

Better customer understanding

Targeted marketing strategies

Improved business decision-making

9. Future Improvements

Customer Lifetime Value (CLV) prediction -> Stored in Feature_Enginnering.ipynb

Real-time segmentation pipeline  -> Stored in Feature_Enginnering.ipynb