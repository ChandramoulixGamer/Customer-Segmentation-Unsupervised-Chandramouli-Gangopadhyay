import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Customer Segmentation", layout="wide")
st.title("Customer Dashboard")
st.write(
    "Analyze customer behavior, segment audiences, and generate data-driven"
    " marketing insights."
)

st.sidebar.header("Input & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
st.caption("Built with AI, ML and Strategy by Chandramouli (◕‿◕)")
st.divider()


def generate_cluster_persona(cluster_means, overall_means, cluster_size, total_size):
    pct_size = (cluster_size / total_size) * 100

    # Avoid zero division when comparing with population
    safe_overall = overall_means.replace(0, 1e-6)
    relative_diff = (cluster_means - safe_overall) / safe_overall.abs()

    highest_feat = relative_diff.idxmax()
    lowest_feat = relative_diff.idxmin()
    max_dev = relative_diff[highest_feat]
    min_dev = relative_diff[lowest_feat]

    monetary_cols = [
        c
        for c in cluster_means.index
        if any(
            k in c.lower()
            for k in ["spend", "amount", "monetary", "income", "revenue", "purchase"]
        )
    ]

    if monetary_cols:
        target_m = monetary_cols[0]
        m_diff = relative_diff[target_m]

        if m_diff > 0.20:
            persona = "High-Value Champions"
            strategy = (
                f"Elevated {target_m} (+{m_diff*100:.1f}% vs avg). "
                "Retain with exclusive VIP perks, priority tier access, and concierge support."
            )
        elif m_diff < -0.20:
            persona = "Budget / At-Risk Shoppers"
            strategy = (
                f"Under-indexing in {target_m} ({m_diff*100:.1f}% vs avg). "
                "Engage with price-drop alerts, entry bundles, and high-discount promotions."
            )
        else:
            persona = "Core Steady Baseline"
            strategy = (
                f"Balanced activity across {target_m}. Maintain engagement with standard "
                "seasonal campaigns and cross-sell nudges."
            )
    else:
        if max_dev > 0.15:
            persona = f"High-{highest_feat.title().replace('_', ' ')} Group"
            strategy = (
                f"Significantly elevated {highest_feat} (+{max_dev*100:.1f}% vs baseline). "
                f"Deploy personalized loyalty incentives focusing on {highest_feat}."
            )
        elif min_dev < -0.15:
            persona = f"Low-{lowest_feat.title().replace('_', ' ')} Group"
            strategy = (
                f"Lags in {lowest_feat} ({min_dev*100:.1f}% vs baseline). "
                f"Implement re-engagement flow and friction-reduction incentives for {lowest_feat}."
            )
        else:
            persona = "Moderate Baseline Cohort"
            strategy = "Average metric distribution. Target with baseline newsletters and product discovery alerts."

    return persona, strategy, pct_size


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    st.subheader("1. Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # Extract numerical columns
    raw_numeric = df.select_dtypes(include=["number"])

    if raw_numeric.shape[1] < 2:
        st.error("The uploaded CSV must contain at least 2 numeric columns.")
        st.stop()

    # Pre-select valid non-ID features
    id_like = [
        c
        for c in raw_numeric.columns
        if any(w in c.lower() for w in ["id", "index", "unnamed"])
    ]
    default_features = [c for c in raw_numeric.columns if c not in id_like]

    # Fallback if every numeric column was detected as ID
    if len(default_features) < 2:
        default_features = list(raw_numeric.columns[:2])

    selected_features = st.sidebar.multiselect(
        "Features for Clustering",
        options=list(raw_numeric.columns),
        default=default_features,
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 numerical features to run clustering.")
        st.stop()

    # Clean numeric data
    numeric_df = raw_numeric[selected_features].copy()
    numeric_df = numeric_df.fillna(numeric_df.mean())

    if len(numeric_df) < 4:
        st.error("Dataset needs at least 4 rows to compute clusters.")
        st.stop()

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Dynamic K selection bounded safely
    max_k_limit = min(8, len(numeric_df) - 1)
    k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=max_k_limit, value=min(3, max_k_limit))

    # Run Model
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled_data)
    df["Cluster"] = clusters

    # Safe Silhouette Calculation
    try:
        score = silhouette_score(scaled_data, clusters)
        st.sidebar.metric("Silhouette Quality Score", f"{score:.3f}")
    except Exception:
        st.sidebar.write("Silhouette Score: N/A")

    # Aggregate metrics
    cluster_profile = numeric_df.groupby(clusters).mean()
    overall_profile = numeric_df.mean()
    cluster_counts = pd.Series(clusters).value_counts()

    persona_map = {}
    strategy_map = {}
    metrics_summary = []

    for c_id in sorted(cluster_profile.index):
        p_name, p_strat, p_pct = generate_cluster_persona(
            cluster_means=cluster_profile.loc[c_id],
            overall_means=overall_profile,
            cluster_size=cluster_counts[c_id],
            total_size=len(df),
        )
        persona_map[c_id] = f"Cluster {c_id}: {p_name}"
        strategy_map[c_id] = p_strat

        summary_row = {"Cluster": f"Cluster {c_id}: {p_name}", "Size (%)": f"{p_pct:.1f}%"}
        for col in selected_features:
            summary_row[col] = round(float(cluster_profile.loc[c_id, col]), 2)
        metrics_summary.append(summary_row)

    df["Customer Segment"] = df["Cluster"].map(persona_map)
    df["Recommended Action"] = df["Cluster"].map(strategy_map)

    # Section 2: Strategy Recommendations
    st.divider()
    st.subheader("2. Metric-Driven Strategy & Recommendations")
    rec_table = (
        df[["Customer Segment", "Recommended Action"]]
        .drop_duplicates()
        .sort_values(by="Customer Segment")
        .reset_index(drop=True)
    )
    st.dataframe(rec_table, use_container_width=True)

    # Section 3: Metric Baseline
    st.subheader("3. Cluster Feature Averages vs Baseline")
    st.dataframe(pd.DataFrame(metrics_summary), use_container_width=True)

    # Section 4: Visualizations
    st.divider()
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("**Feature Correlation Matrix**")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=False,
            ax=ax_corr,
        )
        st.pyplot(fig_corr)

    with v_col2:
        st.markdown("**2D Projection (PCA)**")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
        ax_pca.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=clusters,
            cmap="tab10",
            alpha=0.7,
            edgecolor="k",
            s=40,
        )
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        st.pyplot(fig_pca)

    # Section 5: Export
    st.divider()
    st.subheader("4. Download Processed Cohorts")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Segmented CSV",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv",
    )
else:
    st.info("Awaiting CSV file upload from the sidebar to generate segment metrics.")
