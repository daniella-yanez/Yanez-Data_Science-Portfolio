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
st.set_page_config(page_title="ML Unsupervised App", layout="wide")
st.title("Unsupervised Machine Learning App")

# -----------------------------------------------
# Sidebar: Data Upload and User Inputs
# -----------------------------------------------
with st.sidebar:
    st.header("Upload and Select Options")
    use_sample = st.checkbox("Use Sample Dataset (Country Data)", value=False)
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

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
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Custom dataset uploaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_sample:
    try:
        file_path = os.path.join("data", "Country-data.csv")
        data = pd.read_csv(file_path)
        st.success("Sample dataset loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load sample dataset: {e}")

# -----------------------------------------------
# Main Logic
# -----------------------------------------------
if data is not None:
    data.columns = [col.strip().lower() for col in data.columns]
    numeric_data = data.select_dtypes(include=np.number)

    if numeric_data.shape[1] == 0:
        st.error("No numeric features found in the dataset.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Clustering
        if clustering_method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            cluster_labels = model.fit_predict(X_scaled)

        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        data['cluster'] = cluster_labels

        # PCA Plot
        st.subheader("PCA Scatter Plot")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, edgecolors='k')
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA Clustering Visualization ({clustering_method})")
        st.pyplot(fig)

        st.markdown(f"**Silhouette Score:** {silhouette_avg:.2f}")

        # World Map Visualization
        st.subheader("World Map Visualization (if 'country' column present)")
        st.write("Columns in dataset:", list(data.columns))
        if 'country' in data.columns:
            if PLOTLY_AVAILABLE:
                try:
                    fig_map = px.choropleth(data, locations='country', locationmode='country names',
                                            color='cluster', title='Country Clusters', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_map)
                except Exception as e:
                    st.error(f"Failed to generate choropleth: {e}")
            else:
                st.warning("Plotly not installed. Install it to see the choropleth map.")
        else:
            st.info("Add a 'country' column to your data for map visualization.")

        # Elbow Plot
        if clustering_method == "K-Means":
            st.subheader("K-Means Elbow Plot")
            distortions = []
            K_range = range(2, 11)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(X_scaled)
                distortions.append(km.inertia_)
            fig, ax = plt.subplots()
            ax.plot(K_range, distortions, 'bo-')
            ax.set_xlabel('k')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method For Optimal k')
            st.pyplot(fig)

        # Dendrogram
        if clustering_method == "Hierarchical":
            st.subheader("Hierarchical Dendrogram")
            Z = linkage(X_scaled, method=linkage_method)
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, labels=data.index.to_numpy(), ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Index")
            ax.set_ylabel("Distance")
            st.pyplot(fig)

        # Final Clustered Output
        st.subheader("Clustered Data Preview")
        st.dataframe(data[['cluster'] + [col for col in data.columns if col != 'cluster']].head())
else:
    st.info("Please upload a dataset or select the sample to begin.")
