# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Wholesale Customers Clustering",
    page_icon="🛒",
    layout="wide"
)

# Title and description
st.title("🛒 Wholesale Customers Clustering Analysis")
st.markdown("""
This app performs clustering analysis on wholesale customer data to identify distinct customer segments 
based on their spending patterns across different product categories.
""")

# Sidebar
st.sidebar.header("Configuration")

# Function to load data
@st.cache_data
def load_data():
    # Check if file exists in current directory
    if os.path.exists("Wholesale customers data.csv"):
        df = pd.read_csv("Wholesale customers data.csv")
        return df
    else:
        return None

# Upload data option
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()
    if df is None:
        st.warning("⚠️ Please upload the Wholesale customers data.csv file to begin.")
        st.info("You can download the dataset from the UCI Machine Learning Repository.")
        st.stop()

# Initialize session state
if 'X_scaled' not in st.session_state:
    st.session_state['X_scaled'] = None
if 'X_pca' not in st.session_state:
    st.session_state['X_pca'] = None
if 'labels' not in st.session_state:
    st.session_state['labels'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = None

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Exploration", 
    "📈 Preprocessing", 
    "🔍 Clustering Models", 
    "📉 Visualizations",
    "💾 Model Export"
])

with tab1:
    st.header("Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(10))
        
    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
    
    # Data info
    st.subheader("Data Information")
    buffer = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(buffer)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f')
    plt.title("Feature Correlations")
    st.pyplot(fig_corr)
    plt.close()
    
    # Feature distributions - Fixed version
    st.subheader("Feature Distributions")
    
    # Create individual histograms instead of subplots to avoid the error
    cols = df.columns.tolist()
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    for i in range(0, len(cols), n_cols):
        row_cols = st.columns(n_cols)
        for j, col_name in enumerate(cols[i:i+n_cols]):
            with row_cols[j]:
                fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
                ax_dist.hist(df[col_name], bins=20, edgecolor='black', alpha=0.7)
                ax_dist.set_title(col_name)
                ax_dist.set_xlabel("Value")
                ax_dist.set_ylabel("Frequency")
                st.pyplot(fig_dist)
                plt.close()

with tab2:
    st.header("Data Preprocessing")
    
    st.subheader("Feature Selection")
    
    # Feature selection options
    all_features = df.columns.tolist()
    default_features = [f for f in all_features if f not in ['Channel', 'Region']]
    
    selected_features = st.multiselect(
        "Select features for clustering",
        options=all_features,
        default=default_features
    )
    
    if selected_features:
        X = df[selected_features]
        
        st.subheader("Scaling Options")
        scaling_method = st.radio(
            "Select scaling method",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        X_scaled = scaler.fit_transform(X)
        st.session_state['X_scaled'] = X_scaled
        st.session_state['selected_features'] = selected_features
        st.session_state['scaler'] = scaler
        
        st.success(f"✅ Data scaled successfully! Shape: {X_scaled.shape}")
        
        # PCA projection
        st.subheader("PCA Dimensionality Reduction")
        n_components = st.slider("Number of PCA components", 2, min(10, len(selected_features)), 2)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        st.session_state['X_pca'] = X_pca
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        
        fig_var = px.bar(
            x=[f"PC{i+1}" for i in range(n_components)],
            y=explained_var,
            title="Explained Variance by Principal Components",
            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
        st.write(f"**Total explained variance:** {explained_var.sum():.2%}")
        
        # Show PCA scatter plot
        fig_pca = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            title="PCA Projection (Unclustered Data)",
            labels={'x': 'PC1', 'y': 'PC2'}
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("⚠️ Please select at least one feature for clustering.")

with tab3:
    st.header("Clustering Models")
    
    if st.session_state['X_scaled'] is None:
        st.warning("⚠️ Please complete the preprocessing step first.")
        st.stop()
    
    X_scaled = st.session_state['X_scaled']
    X_pca = st.session_state['X_pca']
    
    # Model selection
    model_type = st.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"]
    )
    
    # Model parameters
    st.subheader("Model Parameters")
    
    # Elbow method for K-Means
    if model_type == "K-Means" and st.checkbox("Show Elbow Method"):
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_scaled)
            inertia.append(kmeans_temp.inertia_)
        
        fig_elbow = px.line(
            x=list(k_range), y=inertia,
            markers=True,
            title="Elbow Method for Optimal k",
            labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Model parameters based on selection
    if model_type == "K-Means":
        n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
        
        if st.button("Run K-Means"):
            with st.spinner("Running K-Means clustering..."):
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                st.session_state['model'] = model
                st.session_state['model_name'] = "K-Means"
                
    elif model_type == "Agglomerative Clustering":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
        
        if st.button("Run Agglomerative Clustering"):
            with st.spinner("Running Hierarchical clustering..."):
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                labels = model.fit_predict(X_scaled)
                st.session_state['model'] = model
                st.session_state['model_name'] = "Agglomerative"
                
    elif model_type == "DBSCAN":
        eps = st.slider("Epsilon (ε)", 0.1, 2.0, 0.5)
        min_samples = st.slider("Minimum samples", 2, 20, 5)
        
        if st.button("Run DBSCAN"):
            with st.spinner("Running DBSCAN clustering..."):
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
                st.session_state['model'] = model
                st.session_state['model_name'] = "DBSCAN"
                
    else:  # Gaussian Mixture
        n_components = st.slider("Number of components", 2, 10, 3)
        covariance_type = st.selectbox("Covariance type", ["full", "tied", "diag", "spherical"])
        
        if st.button("Run Gaussian Mixture"):
            with st.spinner("Running Gaussian Mixture Model..."):
                model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
                labels = model.fit_predict(X_scaled)
                st.session_state['model'] = model
                st.session_state['model_name'] = "GMM"
    
    # Display results if model has been run
    if 'labels' in locals():
        st.session_state['labels'] = labels
        
        # Calculate silhouette score
        unique_labels = len(set(labels))
        if unique_labels > 1 and unique_labels < len(X_scaled):
            sil_score = silhouette_score(X_scaled, labels)
            st.metric("Silhouette Score", f"{sil_score:.3f}")
            
            # Color mapping for noise points in DBSCAN
            if -1 in labels:
                st.info(f"ℹ️ DBSCAN found {list(labels).count(-1)} noise points (labeled as -1)")
        else:
            st.warning("⚠️ Silhouette score cannot be calculated (only one cluster or all points are noise).")
        
        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        fig_dist = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            title="Cluster Distribution",
            labels={'x': 'Cluster', 'y': 'Count'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # 2D visualization
        st.subheader("Cluster Visualization (PCA)")
        
        # Convert labels to string for categorical coloring
        labels_str = labels.astype(str)
        
        fig_cluster = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=labels_str,
            title=f"{model_type} Clustering Results (PCA Projection)",
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

with tab4:
    st.header("Advanced Visualizations")
    
    if st.session_state['X_scaled'] is None or st.session_state['labels'] is None:
        st.warning("⚠️ Please run a clustering model first.")
        st.stop()
    
    X = df[st.session_state['selected_features']]
    labels = st.session_state['labels']
    
    # Parallel coordinates plot
    st.subheader("Parallel Coordinates Plot")
    
    df_plot = X.copy()
    df_plot['Cluster'] = labels.astype(str)
    
    fig_parallel = px.parallel_coordinates(
        df_plot,
        color='Cluster',
        dimensions=st.session_state['selected_features'],
        color_continuous_scale=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig_parallel, use_container_width=True)
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    
    # Calculate mean values per cluster
    cluster_profiles = X.copy()
    cluster_profiles['Cluster'] = labels
    profiles = cluster_profiles.groupby('Cluster').mean()
    
    # Display as dataframe
    st.dataframe(profiles.style.highlight_max(axis=0))
    
    # Heatmap of profiles
    fig_heatmap = px.imshow(
        profiles.T,
        text_auto='.0f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Cluster Profiles (Mean Values)",
        labels={'x': 'Cluster', 'y': 'Features'}
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Feature distributions by cluster
    st.subheader("Feature Distributions by Cluster")
    
    selected_feature = st.selectbox("Select feature to visualize", st.session_state['selected_features'])
    
    df_box = X.copy()
    df_box['Cluster'] = labels.astype(str)
    
    fig_box = px.box(
        df_box,
        x='Cluster', y=selected_feature,
        color='Cluster',
        title=f"Distribution of {selected_feature} by Cluster",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # 3D visualization if enough PCA components
    if st.session_state['X_pca'].shape[1] >= 3:
        st.subheader("3D Cluster Visualization")
        
        X_pca = st.session_state['X_pca']
        
        fig_3d = px.scatter_3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            color=labels.astype(str),
            title="3D PCA Projection",
            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3', 'color': 'Cluster'}
        )
        st.plotly_chart(fig_3d, use_container_width=True)

with tab5:
    st.header("Model Export")
    
    if st.session_state['labels'] is None:
        st.warning("⚠️ Please run a clustering model first.")
        st.stop()
    
    st.subheader("Download Results")
    
    # Prepare results dataframe
    results_df = df.copy()
    results_df['Cluster'] = st.session_state['labels']
    
    # CSV download
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data with Cluster Labels (CSV)",
        data=csv,
        file_name="clustering_results.csv",
        mime="text/csv"
    )
    
    # Save model if available
    if st.session_state['model'] is not None and st.button("Save Model for Deployment"):
        try:
            # Save the model and scaler
            joblib.dump(st.session_state['model'], "trained_model.pkl")
            joblib.dump(st.session_state['scaler'], "scaler.pkl")
            st.success("✅ Model and scaler saved successfully!")
            
            # Also save feature names
            with open("features.txt", "w") as f:
                f.write(",".join(st.session_state['selected_features']))
            
            st.info("Model files: trained_model.pkl, scaler.pkl, features.txt")
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    st.subheader("Cluster Summary Statistics")
    
    # Summary statistics by cluster
    summary = results_df.groupby('Cluster').agg(['mean', 'std', 'count']).round(2)
    st.dataframe(summary)
    
    # Business insights
    st.subheader("Business Insights")
    
    # Simple insights based on cluster characteristics
    cluster_means = results_df.groupby('Cluster')[st.session_state['selected_features']].mean()
    
    for cluster in cluster_means.index:
        if cluster == -1:  # Skip noise points for DBSCAN
            continue
            
        with st.expander(f"**Cluster {cluster}**"):
            # Find top spending category
            top_category = cluster_means.loc[cluster].idxmax()
            top_value = cluster_means.loc[cluster].max()
            
            st.markdown(f"- **Highest spending:** {top_category} (${top_value:.2f} average)")
            
            # Find lowest spending category
            bottom_category = cluster_means.loc[cluster].idxmin()
            bottom_value = cluster_means.loc[cluster].min()
            
            st.markdown(f"- **Lowest spending:** {bottom_category} (${bottom_value:.2f} average)")
            
            # Cluster size
            cluster_size = len(results_df[results_df['Cluster'] == cluster])
            total_size = len(results_df)
            percentage = (cluster_size / total_size) * 100
            
            st.markdown(f"- **Size:** {cluster_size} customers ({percentage:.1f}% of total)")
            
            # Key characteristics
            st.markdown("**Key characteristics:**")
            for feature in st.session_state['selected_features']:
                feature_mean = cluster_means.loc[cluster, feature]
                overall_mean = df[feature].mean()
                comparison = "above" if feature_mean > overall_mean else "below"
                st.markdown(f"  - {feature}: {feature_mean:.0f} ({comparison} overall average of {overall_mean:.0f})")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app performs customer segmentation using various clustering algorithms.
    
    **Features:**
    - Data exploration
    - Multiple clustering algorithms
    - Interactive visualizations
    - Model export
    
    **Algorithms:**
    - K-Means
    - Hierarchical Clustering
    - DBSCAN
    - Gaussian Mixture Models
    """
)

# Add instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### How to use")
st.sidebar.markdown("""
1. Upload your dataset or use the default
2. Explore data in Tab 1
3. Select features and scale in Tab 2
4. Choose algorithm and run clustering in Tab 3
5. Visualize results in Tab 4
6. Export results in Tab 5
""")
