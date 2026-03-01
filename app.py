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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Wholesale Customers Clustering Comparison",
    page_icon="🛒",
    layout="wide"
)

# Title and description
st.title("🛒 Wholesale Customers Clustering Analysis")
st.markdown("""
This app performs clustering analysis on wholesale customer data and compares **K-Means** vs **Hierarchical Clustering** 
to identify distinct customer segments based on their spending patterns.
""")

# Sidebar
st.sidebar.header("Configuration")

# Function to load data
@st.cache_data
def load_data():
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
if 'kmeans_labels' not in st.session_state:
    st.session_state['kmeans_labels'] = None
if 'hierarchical_labels' not in st.session_state:
    st.session_state['hierarchical_labels'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'kmeans_model' not in st.session_state:
    st.session_state['kmeans_model'] = None
if 'hierarchical_model' not in st.session_state:
    st.session_state['hierarchical_model'] = None
if 'pca' not in st.session_state:
    st.session_state['pca'] = None

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Exploration", 
    "📈 Preprocessing", 
    "🔍 Run Both Algorithms", 
    "📉 Comparison & Visualization",
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
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    cols = df.columns.tolist()
    n_cols = 3
    
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
        st.session_state['pca'] = pca
        
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
    st.header("Run Clustering Algorithms")
    
    if st.session_state['X_scaled'] is None:
        st.warning("⚠️ Please complete the preprocessing step first.")
        st.stop()
    
    X_scaled = st.session_state['X_scaled']
    
    st.subheader("Common Parameters")
    n_clusters = st.slider("Number of clusters (k)", 2, 10, 3, key="common_n_clusters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-Means Clustering")
        kmeans_init = st.selectbox("Initialization method", ["k-means++", "random"], key="kmeans_init")
        kmeans_max_iter = st.number_input("Max iterations", 100, 1000, 300, step=50, key="kmeans_iter")
        
        if st.button("Run K-Means", key="run_kmeans"):
            with st.spinner("Running K-Means clustering..."):
                kmeans_model = KMeans(
                    n_clusters=n_clusters, 
                    init=kmeans_init,
                    max_iter=kmeans_max_iter,
                    random_state=42, 
                    n_init=10
                )
                kmeans_labels = kmeans_model.fit_predict(X_scaled)
                st.session_state['kmeans_model'] = kmeans_model
                st.session_state['kmeans_labels'] = kmeans_labels
                st.success("✅ K-Means completed!")
    
    with col2:
        st.subheader("Hierarchical Clustering")
        linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"], key="hier_linkage")
        
        if st.button("Run Hierarchical", key="run_hierarchical"):
            with st.spinner("Running Hierarchical clustering..."):
                hierarchical_model = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    linkage=linkage_method
                )
                hierarchical_labels = hierarchical_model.fit_predict(X_scaled)
                st.session_state['hierarchical_model'] = hierarchical_model
                st.session_state['hierarchical_labels'] = hierarchical_labels
                st.success("✅ Hierarchical clustering completed!")
    
    # Quick comparison if both are run
    if st.session_state['kmeans_labels'] is not None and st.session_state['hierarchical_labels'] is not None:
        st.subheader("Quick Comparison")
        
        kmeans_labels = st.session_state['kmeans_labels']
        hierarchical_labels = st.session_state['hierarchical_labels']
        
        # Calculate metrics
        ari_score = adjusted_rand_score(kmeans_labels, hierarchical_labels)
        nmi_score = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Adjusted Rand Index", f"{ari_score:.3f}")
        with col2:
            st.metric("NMI Score", f"{nmi_score:.3f}")
        with col3:
            agreement = (kmeans_labels == hierarchical_labels).mean() * 100
            st.metric("Label Agreement", f"{agreement:.1f}%")
        
        st.info("These metrics show how similar the two clustering results are. Higher values indicate more agreement.")

with tab4:
    st.header("Comparison & Visualizations")
    
    if st.session_state['X_scaled'] is None:
        st.warning("⚠️ Please complete the preprocessing step first.")
        st.stop()
    
    X = df[st.session_state['selected_features']]
    X_pca = st.session_state['X_pca']
    
    kmeans_labels = st.session_state['kmeans_labels']
    hierarchical_labels = st.session_state['hierarchical_labels']
    
    if kmeans_labels is None and hierarchical_labels is None:
        st.warning("⚠️ Please run at least one clustering algorithm in Tab 3.")
        st.stop()
    
    # Algorithm selection for display
    display_options = []
    if kmeans_labels is not None:
        display_options.append("K-Means")
    if hierarchical_labels is not None:
        display_options.append("Hierarchical")
    
    selected_display = st.multiselect(
        "Select algorithms to compare",
        options=display_options,
        default=display_options
    )
    
    # Side by side PCA visualizations
    if len(selected_display) > 0:
        st.subheader("PCA Projection Comparison")
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=len(selected_display),
            subplot_titles=selected_display,
            shared_yaxes=True
        )
        
        color_sequences = px.colors.qualitative.Set1
        
        for i, algo in enumerate(selected_display, 1):
            if algo == "K-Means":
                labels = kmeans_labels
            else:
                labels = hierarchical_labels
            
            # Convert labels to strings for categorical coloring
            labels_str = labels.astype(str)
            
            fig.add_trace(
                go.Scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        color=labels_str,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"Point {j}<br>Cluster: {labels[j]}" for j in range(len(X_pca))],
                    name=f'{algo} Clusters',
                    showlegend=False
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            height=500,
            title_text="Clustering Results Comparison"
        )
        fig.update_xaxes(title_text="PC1")
        fig.update_yaxes(title_text="PC2")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster distribution comparison
    if len(selected_display) > 1:
        st.subheader("Cluster Distribution Comparison")
        
        dist_data = []
        if "K-Means" in selected_display:
            kmeans_counts = pd.Series(kmeans_labels).value_counts().sort_index()
            for cluster, count in kmeans_counts.items():
                dist_data.append({"Algorithm": "K-Means", "Cluster": f"Cluster {cluster}", "Count": count})
        
        if "Hierarchical" in selected_display:
            hier_counts = pd.Series(hierarchical_labels).value_counts().sort_index()
            for cluster, count in hier_counts.items():
                dist_data.append({"Algorithm": "Hierarchical", "Cluster": f"Cluster {cluster}", "Count": count})
        
        dist_df = pd.DataFrame(dist_data)
        
        fig_dist_comp = px.bar(
            dist_df,
            x="Cluster",
            y="Count",
            color="Algorithm",
            barmode="group",
            title="Cluster Size Distribution Comparison",
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        st.plotly_chart(fig_dist_comp, use_container_width=True)
    
    # Silhouette Score Comparison
    if len(selected_display) > 1:
        st.subheader("Silhouette Score Comparison")
        
        sil_scores = []
        X_scaled = st.session_state['X_scaled']
        
        if "K-Means" in selected_display and len(set(kmeans_labels)) > 1:
            sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
            sil_scores.append({"Algorithm": "K-Means", "Silhouette Score": sil_kmeans})
        
        if "Hierarchical" in selected_display and len(set(hierarchical_labels)) > 1:
            sil_hier = silhouette_score(X_scaled, hierarchical_labels)
            sil_scores.append({"Algorithm": "Hierarchical", "Silhouette Score": sil_hier})
        
        if sil_scores:
            sil_df = pd.DataFrame(sil_scores)
            
            fig_sil = px.bar(
                sil_df,
                x="Algorithm",
                y="Silhouette Score",
                color="Algorithm",
                title="Silhouette Score Comparison (Higher is Better)",
                range_y=[0, 1],
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig_sil, use_container_width=True)
    
    # Radar Charts Comparison
    st.subheader("Cluster Profiles Comparison")
    
    if "K-Means" in selected_display:
        st.markdown("**K-Means Cluster Profiles**")
        kmeans_profiles = X.copy()
        kmeans_profiles['Cluster'] = kmeans_labels
        kmeans_means = kmeans_profiles.groupby('Cluster').mean()
        
        # Normalize
        kmeans_norm = (kmeans_means - kmeans_means.min()) / (kmeans_means.max() - kmeans_means.min())
        
        fig_kmeans_radar = go.Figure()
        for cluster in kmeans_norm.index:
            fig_kmeans_radar.add_trace(go.Scatterpolar(
                r=kmeans_norm.loc[cluster].values.tolist(),
                theta=kmeans_norm.columns.tolist(),
                fill='toself',
                name=f'Cluster {cluster}'
            ))
        
        fig_kmeans_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_kmeans_radar, use_container_width=True)
    
    if "Hierarchical" in selected_display:
        st.markdown("**Hierarchical Cluster Profiles**")
        hier_profiles = X.copy()
        hier_profiles['Cluster'] = hierarchical_labels
        hier_means = hier_profiles.groupby('Cluster').mean()
        
        # Normalize
        hier_norm = (hier_means - hier_means.min()) / (hier_means.max() - hier_means.min())
        
        fig_hier_radar = go.Figure()
        for cluster in hier_norm.index:
            fig_hier_radar.add_trace(go.Scatterpolar(
                r=hier_norm.loc[cluster].values.tolist(),
                theta=hier_norm.columns.tolist(),
                fill='toself',
                name=f'Cluster {cluster}'
            ))
        
        fig_hier_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_hier_radar, use_container_width=True)
    
    # Feature distributions comparison
    st.subheader("Feature Distribution by Algorithm")
    
    selected_feature = st.selectbox("Select feature to compare", st.session_state['selected_features'])
    
    comp_data = []
    if "K-Means" in selected_display:
        temp_df = X[[selected_feature]].copy()
        temp_df['Algorithm'] = 'K-Means'
        temp_df['Cluster'] = kmeans_labels.astype(str)
        temp_df['Value'] = temp_df[selected_feature]
        comp_data.append(temp_df[['Algorithm', 'Cluster', 'Value']])
    
    if "Hierarchical" in selected_display:
        temp_df = X[[selected_feature]].copy()
        temp_df['Algorithm'] = 'Hierarchical'
        temp_df['Cluster'] = hierarchical_labels.astype(str)
        temp_df['Value'] = temp_df[selected_feature]
        comp_data.append(temp_df[['Algorithm', 'Cluster', 'Value']])
    
    if comp_data:
        comp_df = pd.concat(comp_data, ignore_index=True)
        
        fig_comp_box = px.box(
            comp_df,
            x="Cluster",
            y="Value",
            color="Algorithm",
            facet_col="Algorithm",
            title=f"Distribution of {selected_feature} by Cluster and Algorithm",
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        st.plotly_chart(fig_comp_box, use_container_width=True)
    
    # Confusion Matrix / Agreement Heatmap
    if kmeans_labels is not None and hierarchical_labels is not None:
        st.subheader("Cluster Agreement Heatmap")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(kmeans_labels, hierarchical_labels)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        fig_cm = px.imshow(
            cm_norm,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='Blues',
            title="Cluster Agreement (K-Means vs Hierarchical)",
            labels=dict(x="Hierarchical Clusters", y="K-Means Clusters", color="Proportion")
        )
        fig_cm.update_xaxes(side="bottom")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("""
        **How to interpret:** Each row shows how K-Means clusters map to Hierarchical clusters.
        - Values close to 1 indicate strong agreement between clusters
        - Values spread across multiple columns indicate disagreement
        """)

with tab5:
    st.header("Model Export & Summary")
    
    if st.session_state['X_scaled'] is None:
        st.warning("⚠️ Please complete the preprocessing step first.")
        st.stop()
    
    # Algorithm selection for export
    st.subheader("Select Model to Export")
    
    export_options = []
    if st.session_state['kmeans_labels'] is not None:
        export_options.append("K-Means")
    if st.session_state['hierarchical_labels'] is not None:
        export_options.append("Hierarchical")
    
    if not export_options:
        st.warning("⚠️ Please run at least one clustering algorithm in Tab 3.")
        st.stop()
    
    selected_export = st.radio(
        "Choose algorithm for export",
        options=export_options
    )
    
    # Prepare results based on selection
    results_df = df.copy()
    
    if selected_export == "K-Means":
        results_df['Cluster'] = st.session_state['kmeans_labels']
        model = st.session_state['kmeans_model']
        model_name = "K-Means"
    else:
        results_df['Cluster'] = st.session_state['hierarchical_labels']
        model = st.session_state['hierarchical_model']
        model_name = "Hierarchical"
    
    # Add the other algorithm's labels for comparison if available
    if st.session_state['kmeans_labels'] is not None and selected_export != "K-Means":
        results_df['KMeans_Cluster'] = st.session_state['kmeans_labels']
    if st.session_state['hierarchical_labels'] is not None and selected_export != "Hierarchical":
        results_df['Hierarchical_Cluster'] = st.session_state['hierarchical_labels']
    
    st.subheader("Download Results")
    
    # CSV download
    csv = results_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {model_name} Results with Labels (CSV)",
        data=csv,
        file_name=f"{model_name.lower()}_clustering_results.csv",
        mime="text/csv"
    )
    
    # Save model if available
    if model is not None and st.button(f"Save {model_name} Model for Deployment"):
        try:
            joblib.dump(model, f"{model_name.lower()}_model.pkl")
            joblib.dump(st.session_state['scaler'], "scaler.pkl")
            st.success(f"✅ {model_name} model and scaler saved successfully!")
            
            with open("features.txt", "w") as f:
                f.write(",".join(st.session_state['selected_features']))
            
            st.info(f"Model files: {model_name.lower()}_model.pkl, scaler.pkl, features.txt")
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    # Summary Statistics
    st.subheader(f"{model_name} Cluster Summary Statistics")
    summary = results_df.groupby('Cluster')[st.session_state['selected_features']].agg(['mean', 'std', 'count']).round(2)
    st.dataframe(summary)
    
    # Comparison Summary
    if st.session_state['kmeans_labels'] is not None and st.session_state['hierarchical_labels'] is not None:
        st.subheader("Algorithm Comparison Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**K-Means**")
            st.metric("Number of Clusters", len(set(st.session_state['kmeans_labels'])))
            if len(set(st.session_state['kmeans_labels'])) > 1:
                sil_kmeans = silhouette_score(st.session_state['X_scaled'], st.session_state['kmeans_labels'])
                st.metric("Silhouette Score", f"{sil_kmeans:.3f}")
        
        with col2:
            st.markdown("**Hierarchical**")
            st.metric("Number of Clusters", len(set(st.session_state['hierarchical_labels'])))
            if len(set(st.session_state['hierarchical_labels'])) > 1:
                sil_hier = silhouette_score(st.session_state['X_scaled'], st.session_state['hierarchical_labels'])
                st.metric("Silhouette Score", f"{sil_hier:.3f}")
        
        # Recommendation
        st.subheader("Recommendation")
        
        if len(set(st.session_state['kmeans_labels'])) > 1 and len(set(st.session_state['hierarchical_labels'])) > 1:
            if sil_kmeans > sil_hier:
                st.success(f"✅ **K-Means** performs better based on silhouette score ({sil_kmeans:.3f} vs {sil_hier:.3f})")
            elif sil_hier > sil_kmeans:
                st.success(f"✅ **Hierarchical Clustering** performs better based on silhouette score ({sil_hier:.3f} vs {sil_kmeans:.3f})")
            else:
                st.info("Both algorithms perform similarly")
    
    # Business insights for selected model
    st.subheader(f"Business Insights - {model_name}")
    
    cluster_means = results_df.groupby('Cluster')[st.session_state['selected_features']].mean()
    
    for cluster in cluster_means.index:
        with st.expander(f"**Cluster {cluster}**"):
            top_category = cluster_means.loc[cluster].idxmax()
            top_value = cluster_means.loc[cluster].max()
            st.markdown(f"- **Highest spending:** {top_category} (${top_value:.2f} average)")
            
            bottom_category = cluster_means.loc[cluster].idxmin()
            bottom_value = cluster_means.loc[cluster].min()
            st.markdown(f"- **Lowest spending:** {bottom_category} (${bottom_value:.2f} average)")
            
            cluster_size = len(results_df[results_df['Cluster'] == cluster])
            total_size = len(results_df)
            percentage = (cluster_size / total_size) * 100
            st.markdown(f"- **Size:** {cluster_size} customers ({percentage:.1f}% of total)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app performs customer segmentation and compares **K-Means** vs **Hierarchical Clustering**.
    
    **Features:**
    - Data exploration
    - Run both algorithms simultaneously
    - Side-by-side comparison
    - Agreement metrics and visualizations
    - Export results from either algorithm
    
    **Comparison Metrics:**
    - Silhouette Score
    - Adjusted Rand Index
    - NMI Score
    - Cluster agreement heatmaps
    """
)

# Add instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### How to use")
st.sidebar.markdown("""
1. Upload dataset or use default
2. Explore data in Tab 1
3. Select features and scale in Tab 2
4. Run both algorithms in Tab 3
5. Compare results in Tab 4
6. Export preferred model in Tab 5
""")

# Display status
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")

if st.session_state['kmeans_labels'] is not None:
    st.sidebar.success("✅ K-Means: Ready")
else:
    st.sidebar.info("⏳ K-Means: Not run")

if st.session_state['hierarchical_labels'] is not None:
    st.sidebar.success("✅ Hierarchical: Ready")
else:
    st.sidebar.info("⏳ Hierarchical: Not run")
