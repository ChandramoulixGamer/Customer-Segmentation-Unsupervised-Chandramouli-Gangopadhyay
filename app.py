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
    """Dynamically analyzes metric deviations from dataset averages to label personas

    and prescribe targeted marketing actions.
    """
    pct_size = (cluster_size / total_size) * 100

    # Calculate z-score style relative difference: (Cluster Mean - Global Mean) / Global Std
    # Find standout strong and weak dimensions for this cluster
    relative_diff = (cluster_means - overall_means) / (overall_means.abs() + 1e-6)

    highest_feat = relative_diff.idxmax()
    lowest_feat = relative_diff.idxmin()
    max_dev = relative_diff[highest_feat]
    min_dev = relative_diff[lowest_feat]

    # Rule-based persona matching across common feature naming patterns
    col_str = " ".join(cluster_means.index).lower()

    # Pattern 1: Monetary / Spending / Income features present
    monetary_cols = [
        c
        for c in cluster_means.index
        if any(
            k in c.lower()
            for k in ["spend", "amount", "monetary", "income", "revenue", "purchase"]
        )
    ]
    recency_freq_cols = [
        c
        for c in cluster_means.index
        if any(
            k in c.lower()
            for k in ["recency", "frequency", "orders", "visits", "tenure"]
        )
    ]

    if monetary_cols:
        target_m = monetary_cols[0]
        m_diff = relative_diff[target_m]

        if m_diff > 0.25:
            persona = "High-Value Champions"
            strategy = (
                f"Top drivers in {target_m} (+{m_diff*100:.1f}% vs avg). "
                "Retain with exclusive VIP loyalty tiers, early feature/product access, and dedicated support."
            )
        elif m_diff < -0.25:
            persona = "Budget / At-Risk Shoppers"
            strategy = (
                f"Under-indexing in {target_m} ({m_diff*100:.1f}% vs avg). "
                "Engage with entry-level bundles, discount coupons, and low-friction starter incentives."
            )
        else:
            persona = "Core Steady Baseline"
            strategy = (
                f"Moderate behavior across key metrics. Nurture with routine cross-sell recommendations "
                "and seasonal reactivation triggers."
            )
    else:
        # Generic fallback based on highest & lowest mathematical variance
        if max_dev > 0.20:
            persona = f"High-{highest_feat.title().replace('_', ' ')} Group"
            strategy = (
                f"Significantly elevated {highest_feat} (+{max_dev*100:.1f}% vs population). "
                f"Target campaigns centered around {highest_feat} optimization and personalized upgrades."
            )
        elif min_dev < -0.20:
            persona = f"Low-{lowest_feat.title().replace('_', ' ')} Group"
            strategy = (
                f"Lags significantly in {lowest_feat} ({min_dev*100:.1f}% vs population). "
                f"Deploy incentive schemes to revive participation and remove friction in {lowest_feat}."
            )
        else:
            persona = "Moderate Engagement Cohort"
            strategy = (
                "Evenly distributed across metrics. Standardize promotional messaging and A/B test re-engagement."
            )

    return persona, strategy, pct_size


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)
    st.write(
        f"**Dataset dimensions:** {df.shape[0]} rows × {df.shape[1]} columns"
    )

    # Filter out common ID columns to prevent distance distortion
    raw_numeric = df.select_dtypes(include=["int64", "float64"])
    id_like_cols = [
        c
        for c in raw_numeric.columns
        if any(w in c.lower() for w in ["id", "index", "unnamed", "customer_id"])
    ]

    selected_features = st.sidebar.multiselect(
        "Features for Clustering",
        options=list(raw_numeric.columns),
        default=[c for c in raw_numeric.columns if c not in id_like_cols],
    )

    if len(selected_features) < 2:
        st.error(
            "Please select at least 2 numerical features for clustering."
        )
        st.stop()

    numeric_df = raw_numeric[selected_features].copy()
    numeric_df = numeric_df.fillna(numeric_df.mean())

    if len(numeric_df) < 3 or numeric_df.nunique().sum() <= len(
        numeric_df.columns
    ):
        st.error("Insufficient variation or data points for clustering.")
        st.stop()

    # Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Model parameters & Optimal K Evaluation
    max_k = min(8, len(numeric_df) - 1)
    col_k1, col_k2 = st.sidebar.columns(2)
    k = col_k1.slider(
        "K Clusters", min_value=2, max_value=max_k, value=min(3, max_k)
    )

    # Run KMeans
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled_data)
    df["Cluster"] = clusters

    # Evaluation Metric
    sil_score = silhouette_score(scaled_data, clusters)
    st.sidebar.metric("Silhouette Score", f"{sil_score:.3f}")

    # Compute Global & Cluster Aggregates
    cluster_profile = numeric_df.groupby(clusters).mean()
    overall_profile = numeric_df.mean()
    cluster_counts = pd.Series(clusters).value_counts()

    # Build Metric-Driven Strategy Mappings
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
            summary_row[col] = round(cluster_profile.loc[c_id, col], 2)
        metrics_summary.append(summary_row)

    df["Customer Segment"] = df["Cluster"].map(persona_map)
    df["Recommended Action"] = df["Cluster"].map(strategy_map)

    # UI: Metric Summaries & Recommendations
    st.divider()
    st.subheader("2. Metric-Driven Strategy & Recommendations")

    rec_table = (
        df[["Customer Segment", "Recommended Action"]]
        .drop_duplicates()
        .sort_values(by="Customer Segment")
        .reset_index(drop=True)
    )
    st.dataframe(rec_table, use_container_width=True)

    st.subheader("3. Cluster Feature Averages vs Baseline")
    st.dataframe(pd.DataFrame(metrics_summary), use_container_width=True)

    # Visualizations
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
        scatter = ax_pca.scatter(
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

    # Export
    st.divider()
    st.subheader("4. Download Processed Cohorts")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Segmented CSV",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv",
    )
