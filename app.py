import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Customer Segmentation", layout="wide")
st.title("Customer Dashboard")
st.write(
    "Analyze customer behavior, segment audiences, and generate AI-recommended"
    " marketing insights (◕‿◕)"
)

st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
st.caption("Built with AI, ML and Strategy by Chandramouli (◕‿◕)")
st.divider()

if uploaded_file:
  df = pd.read_csv(uploaded_file)

  st.write("Dataset Preview:")
  st.dataframe(df.head())
  st.write(
      f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
  )
  st.divider()

  # Preprocess numeric data
  numeric_df = df.select_dtypes(include=["int64", "float64"])
  numeric_df = numeric_df.fillna(numeric_df.mean())

  # Validation Checks
  if numeric_df.empty:
    st.error(
        "No valid numeric data found for clustering. Please upload a dataset"
        " with numeric columns."
    )
    st.stop()

  if len(numeric_df.columns) < 2 or len(numeric_df) < 3:
    st.error(
        "Dataset needs at least 2 numeric columns and 3 rows for clustering."
    )
    st.stop()

  if numeric_df.nunique().sum() <= len(numeric_df.columns):
    st.error("Dataset lacks enough variation for meaningful clustering.")
    st.stop()

  st.write("Missing Values Per Column:")
  st.write(df.isnull().sum())
  st.divider()

  # Correlation Heatmap
  st.subheader("Correlation Heatmap")
  fig2, ax2 = plt.subplots(figsize=(10, 6))
  sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
  st.pyplot(fig2)
  st.divider()

  # Scaler & Clustering Setup
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(numeric_df)

  # Dynamic cluster selector bounded by available samples
  max_k = min(10, len(numeric_df))
  k = st.sidebar.slider(
      "Select Number of Clusters",
      min_value=2,
      max_value=max_k,
      value=min(3, max_k),
  )

  model = KMeans(n_clusters=k, random_state=42)
  clusters = model.fit_predict(scaled_data)
  df["Cluster"] = clusters

  # Dynamic segment naming and recommendations strategy for any K value
  base_strategies = [
      "Offer targeted discounts, entry-level bundles, and budget incentives.",
      "Send limited-time offers and automated re-engagement campaigns.",
      "Provide personalized product recommendations based on past purchases.",
      "Focus on retention with exclusive perks and loyalty points.",
      (
          "Offer VIP memberships, premium rewards, and priority early access"
          " deals."
      ),
      "Engage via targeted cross-selling and product discovery guides.",
      "Deliver bespoke high-touch offers and direct support channels.",
      "Re-engage with feedback surveys and special return incentives.",
      "Nurture with educational content and value-add product tips.",
      "Reward with personalized anniversary deals and milestone gifts.",
  ]

  # Generate mappings dynamically based on chosen k
  cluster_labels = {i: f"Customer Segment {i+1}" for i in range(k)}
  recommendations = {
      f"Customer Segment {i+1}": base_strategies[i % len(base_strategies)]
      for i in range(k)
  }

  df["Customer Type"] = df["Cluster"].map(cluster_labels)
  df["Marketing Strategy"] = df["Customer Type"].map(recommendations)

  # PCA Visualization
  pca = PCA(n_components=2)
  reduced_data = pca.fit_transform(scaled_data)

  fig, ax = plt.subplots(figsize=(8, 5))
  scatter = ax.scatter(
      reduced_data[:, 0],
      reduced_data[:, 1],
      c=clusters,
      cmap="viridis",
      alpha=0.8,
  )
  ax.set_title("Segmented Audiences (PCA Reduced)")
  ax.set_xlabel("PCA Component 1")
  ax.set_ylabel("PCA Component 2")
  st.pyplot(fig)
  st.divider()

  # Output Reports
  st.subheader("Segment-Based Marketing Recommendations")
  st.dataframe(
      df[["Customer Type", "Marketing Strategy"]].drop_duplicates(),
      use_container_width=True,
  )
  st.divider()

  st.write("Full Segmented Customer Dataset:")
  st.dataframe(df, use_container_width=True)

  csv = df.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="Export AI Customer Segmentation Report",
      data=csv,
      file_name="segmented_customers.csv",
      mime="text/csv",
  )
