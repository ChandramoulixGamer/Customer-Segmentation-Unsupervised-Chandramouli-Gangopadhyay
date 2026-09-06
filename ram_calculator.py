import sys
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Streamlit RAM & File Budget", layout="centered")

st.title("Streamlit 1 GB RAM & File Budget Checker")
st.write(
    "Calculates real-time DataFrame memory footprint and projections against"
    " Streamlit Cloud's 1 GB container limit."
)

MAX_RAM_MB = 1024.0  # 1 GB
SAFE_RAM_THRESHOLD_MB = 700.0  # Safe ceiling leaving room for Python runtime, OS, and ML models

# Mode selector
mode = st.radio("Choose Input Mode:", ["Simulate by File Size", "Upload Real CSV"], horizontal=True)

if mode == "Simulate by File Size":
    st.subheader("Memory Consumption Estimator")
    file_size_mb = st.slider("Select Uncompressed CSV Size (MB):", min_value=1.0, max_value=250.0, value=30.0, step=1.0)
    
    # In Pandas, parsed CSVs typically expand 3x to 5x due to 64-bit data structures and string metadata
    estimated_df_ram = file_size_mb * 4.0
    total_projected_ram = estimated_df_ram + 150.0  # ~150 MB baseline for Streamlit + runtime libraries
    
    pct_used = min(100.0, (total_projected_ram / MAX_RAM_MB) * 100)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw CSV File Size", f"{file_size_mb:.1f} MB")
    col2.metric("Projected RAM in Pandas", f"{estimated_df_ram:.1f} MB", help="Assumes typical 4x expansion for tabular records")
    col3.metric("Total Container RAM", f"{total_projected_ram:.1f} / 1024 MB")
    
    st.progress(pct_used / 100.0)
    
    if total_projected_ram > MAX_RAM_MB:
        st.error(f"⚠️ **High Risk of OOM Crash:** Exceeds 1 GB by {total_projected_ram - MAX_RAM_MB:.1f} MB. Processing spatial matrices (KMeans/PCA) will trigger an out-of-memory abort.")
    elif total_projected_ram > SAFE_RAM_THRESHOLD_MB:
        st.warning(f"⚠️ **Caution:** Leaves less than {MAX_RAM_MB - total_projected_ram:.1f} MB of free headroom. Aggregation or subsampling recommended.")
    else:
        st.success(f"✅ **Safe Zone:** App has {MAX_RAM_MB - total_projected_ram:.1f} MB of free memory remaining for clustering and visualizations.")

else:
    st.subheader("Real-time File Profiler")
    uploaded = st.file_uploader("Upload CSV to inspect exact memory consumption:", type=["csv"])
    
    if uploaded:
        raw_size_mb = uploaded.size / (1024 * 1024)
        
        # Load and compute true memory
        try:
            df = pd.read_csv(uploaded)
            exact_ram_bytes = df.memory_usage(deep=True).sum()
            exact_ram_mb = exact_ram_bytes / (1024 * 1024)
            expansion_factor = exact_ram_mb / max(0.001, raw_size_mb)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Disk Size", f"{raw_size_mb:.2f} MB")
            c2.metric("In-Memory RAM", f"{exact_ram_mb:.2f} MB")
            c3.metric("Expansion Factor", f"{expansion_factor:.1f}x")
            c4.metric("Row Count", f"{len(df):,}")
            
            runtime_headroom = MAX_RAM_MB - (exact_ram_mb + 150.0)
            st.write(f"**Available Container RAM Remaining for ML Tasks:** `{max(0.0, runtime_headroom):.1f} MB`")
            st.progress(min(1.0, (exact_ram_mb + 150.0) / MAX_RAM_MB))
            
            # Show breakdown per column
            with st.expander("Detailed Memory Usage by Column"):
                col_mem = (df.memory_usage(deep=True) / (1024 * 1024)).reset_index()
                col_mem.columns = ["Column", "Memory (MB)"]
                st.dataframe(col_mem.sort_values(by="Memory (MB)", ascending=False), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
