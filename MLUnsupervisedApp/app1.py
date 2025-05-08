import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import os

# Optional: Import Plotly if available
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False

# Optional: Import KaggleHub if available
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ModuleNotFoundError:
    KAGGLEHUB_AVAILABLE = False

# -----------------------------------------------
# Streamlit App Setup
# -----------------------------------------------
st.set_page_config(layout="wide")
st.title("Interactive Clustering Explorer")

# -----------------------------------------------
# Sidebar: Data Upload and User Inputs
# -----------------------------------------------
with st.sidebar:
    st.header("Upload and Select Options")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    use_sample = st.checkbox("Use Sample Dataset (Country Data)", value=False)

    clustering_method = st.selectbox("Choose clustering method:", ["K-Means", "Hierarchical"])
    n_clusters = st.slider("Number of clusters (k):", 2, 10, 4)

    linkage_method = "ward"
    if clustering_method == "Hierarchical":
        linkage_method = st.selectbox("Linkage method:", ["ward", "single", "complete", "average"])

    n_components = st.slider("# PCA Components for Visualization:", 2, 3, 2)

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------
data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    if KAGGLEHUB_AVAILABLE:
        try:
            path = kagglehub.dataset_download("rohan0301/unsupervised-learning-on-country-data")
            file_path = os.path.join(path, 'Country-data.csv')
            data = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")
    else:
        st.warning("KaggleHub not installed. Please upload your own dataset or install KaggleHub.")

if data is not None:
    numeric_data = data.select_dtypes(include=np.number)

    if numeric_data.shape[1] == 0:
        st.error("No numeric features found in the dataset.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # -----------------------------------------------
        # Preprocessing and Dimensionality Reduction
        # -----------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # -----------------------------------------------
        # Clustering
        # -----------------------------------------------
        if clustering_method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = model.fit_predict(X_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            cluster_labels = model.fit_predict(X_scaled)

        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        data['Cluster'] = cluster_labels

        # -----------------------------------------------
        # PCA Scatter Plot Visualization
        # -----------------------------------------------
        st.subheader("PCA Scatter Plot")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, edgecolors='k')
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA Clustering Visualization ({clustering_method})")
        st.pyplot(fig)

        st.markdown(f"**Silhouette Score:** {silhouette_avg:.2f}")

        # -----------------------------------------------
        # World Map Visualization (if 'country' column exists)
        # -----------------------------------------------
        st.subheader("World Map Visualization (if 'country' column present)")
        if 'country' in data.columns:
            if PLOTLY_AVAILABLE:
                fig_map = px.choropleth(data, locations='country', locationmode='country names',
                                        color='Cluster', title='Country Clusters', color_continuous_scale='Viridis')
                st.plotly_chart(fig_map)
            else:
                st.warning("Plotly not installed. Install it to see the choropleth map.")
        else:
            st.info("Add a 'country' column to your data for map visualization.")

        # -----------------------------------------------
        # Elbow Plot for K-Means
        # -----------------------------------------------
        if clustering_method == "K-Means":
            st.subheader("K-Means Elbow Plot")
            distortions = []
            K_range = range(2, 11)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init='auto')
                km.fit(X_scaled)
                distortions.append(km.inertia_)
            fig, ax = plt.subplots()
            ax.plot(K_range, distortions, 'bo-')
            ax.set_xlabel('k')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method For Optimal k')
            st.pyplot(fig)

        # -----------------------------------------------
        # Dendrogram for Hierarchical Clustering
        # -----------------------------------------------
        if clustering_method == "Hierarchical":
            st.subheader("Hierarchical Dendrogram")
            Z = linkage(X_scaled, method=linkage_method)
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, labels=data.index.to_numpy(), ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Index")
            ax.set_ylabel("Distance")
            st.pyplot(fig)

        # -----------------------------------------------
        # Display Clustered Data
        # -----------------------------------------------
        st.subheader("Clustered Data Preview")
        st.dataframe(data[['Cluster'] + [col for col in data.columns if col != 'Cluster']].head())
else:
    st.info("Please upload a dataset or select a sample to begin.")
