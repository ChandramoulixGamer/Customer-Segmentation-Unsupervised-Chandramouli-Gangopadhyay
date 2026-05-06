import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Customer Segmentation", layout="wide")
st.title("Customer Dashboard")
st.write("Analyze customer behavior, segment audiences, and generate AI-recmmended marketing insights (◕‿◕)")

st.sidebar.header("Input")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
st.caption("Built with AI,ML and Strategy by Chandramouli(◕‿◕)")
st.divider()

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.divider()
    
    
    numeric_df = df.select_dtypes(include=['int64', 'float64']).fillna(df.select_dtypes(include=['int64', 'float64']).mean())
    if numeric_df.empty:
        st.error("No valid numeric data found for clustering. Please upload a dataset with numeric columns.")
    if numeric_df.nunique().sum() <= len(numeric_df.columns):
        st.error("Dataset lacks enough variation for meaningful clustering.")
        st.warning("Clustering works best with customer-like datasets containing multiple numeric behavioral features.")
    st.write("Missing Values Per Column:")
    st.write(df.isnull().sum())
    st.divider()
      #Heatmap
    st.subheader("Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
    st.divider()


    if len(numeric_df.columns) >= 2 and len(numeric_df) > 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        k = st.sidebar.slider("Select Number of Clusters", 2, 10, 3)

        model = KMeans(n_clusters=k, random_state=42)
        clusters = model.fit_predict(scaled_data)
        df["Cluster"] = clusters
        cluster_labels = {
    0: "Budget-Tier Customers",
    1: "Occasional Customers",
    2: "Regular Customers",
    3: "Valuable Customers",
    4: "Premium Customers"
}
        df["Customer Type"] = df["Cluster"].map(cluster_labels).fillna("Other Segment")
        recommendations = {
    "Budget-Tier Customers": "Offer discounts, coupons, and budget bundles.",
    "Occasional Customers": "Send limited-time offers and re-engagement campaigns.",
    "Regular Customers": "Provide personalized product recommendations.",
    "Valuable Customers": "Focus on retention with exclusive perks.",
    "Premium Customers": "Offer VIP memberships, premium loyalty rewards, and early access deals."
}
        
        df["Marketing Strategy"] = df["Customer Type"].map(recommendations)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots()
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
        ax.set_title("Segmented Audiences")
        st.pyplot(fig)
        st.divider()
        st.write("AI-Segmented Customer Profiles with Marketing Insights:")
        st.subheader("Segment-Based Marketing Recommendations")
        st.dataframe(df[["Customer Type", "Marketing Strategy"]].drop_duplicates())
        st.divider()
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export AI Customer Segmentation Report",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
)
        st.subheader("Export Business Intelligence Report")
        
    else:
        st.error("Need at least 2 numeric columns....  Please try again with Valid Data :)")
        
