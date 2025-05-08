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

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------
data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    try:
        file_path = os.path.join("data", "Country-data.csv")
        data = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load bundled sample dataset: {e}")

if data is not None:
    initial_rows = data.shape[0]
    data = data.dropna()
    dropped_rows = initial_rows - data.shape[0]

    if dropped_rows > 0:
        st.warning(f"Dropped {dropped_rows} rows with missing values.")

    data.columns = [col.strip().lower() for col in data.columns]
    numeric_data = data.select_dtypes(include=np.number)

    if numeric_data.shape[1] == 0:
        st.error("No numeric features found in the dataset.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # -----------------------------------------------
        # Preprocessing and PCA
        # -----------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Suggest optimal number of clusters (KMeans with silhouette)
        silhouette_scores = {}
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X_scaled)
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
        suggested_k = max(silhouette_scores, key=silhouette_scores.get)

        # Sidebar controls after suggestion
        with st.sidebar:
            st.header("Clustering Parameters")
            clustering_method = st.selectbox("Choose clustering method:", ["K-Means", "Hierarchical"])
            use_suggestion = st.checkbox(f"Use suggested number of clusters (k={suggested_k})", value=True)
            if use_suggestion:
                n_clusters = suggested_k
            else:
                n_clusters = st.slider("Number of clusters (k):", 2, 10, suggested_k)

            linkage_method = "ward"
            if clustering_method == "Hierarchical":
                linkage_method = st.selectbox("Linkage method:", ["ward", "single", "complete", "average"])

            n_components = st.slider("# PCA Components for Visualization:", 2, 3, 2)

            if clustering_method == "Hierarchical":
                sample_size = st.slider("Sample size for dendrogram (0 = full dataset):", 0, 500, 100)
                truncate_dendrogram = st.checkbox("Truncate dendrogram? (Show only last p merges)")
                truncate_mode = 'lastp' if truncate_dendrogram else None
                p_value = st.slider("# Clusters to display (used if truncating):", 2, 30, 10) if truncate_dendrogram else None

        st.info(f"Suggested number of clusters based on silhouette score: {suggested_k}")

        # Run PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # -----------------------------------------------
        # Clustering
        # -----------------------------------------------
        if clustering_method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            cluster_labels = model.fit_predict(X_scaled)

        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        data['cluster'] = cluster_labels

        # -----------------------------------------------
        # PCA Scatter Plot
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
        # World Map Visualization
        # -----------------------------------------------
        st.subheader("World Map Visualization")

        if "country" in data.columns:
            if PLOTLY_AVAILABLE:
                numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

                if not numeric_columns:
                    st.warning("No numeric columns found for choropleth visualization.")
                else:
                    default_var = "cluster" if "cluster" in numeric_columns else numeric_columns[0]
                    choropleth_variable = st.selectbox(
                        "Choose variable to map:",
                        options=numeric_columns,
                        index=numeric_columns.index(default_var)
                    )

                    try:
                        fig_map = px.choropleth(
                            data_frame=data,
                            locations="country",
                            locationmode="country names",
                            color=choropleth_variable,
                            hover_name="country",
                            hover_data=numeric_columns,
                            color_continuous_scale="Viridis",
                            title=f"Choropleth Map: {choropleth_variable}"
                        )
                        st.plotly_chart(fig_map)
                    except Exception as e:
                        st.error(f"Error generating choropleth map: {e}")
            else:
                st.warning("Plotly not installed. Install it with `pip install plotly` to view the map.")
        else:
            st.info("To show a choropleth map, your dataset must include a column named 'country'.")

        # -----------------------------------------------
        # Elbow Plot for K-Means
        # -----------------------------------------------
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

        # -----------------------------------------------
        # Dendrogram for Hierarchical Clustering
        # -----------------------------------------------
        if clustering_method == "Hierarchical":
            st.subheader("Hierarchical Dendrogram")
            sample_indices = data.index.to_numpy()
            if sample_size > 0 and sample_size < X_scaled.shape[0]:
                sampled_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                X_dendro = X_scaled[sampled_indices]
                sample_indices = data.index[sampled_indices]
            else:
                X_dendro = X_scaled

            Z = linkage(X_dendro, method=linkage_method)
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(
                Z,
                labels=sample_indices,
                ax=ax,
                truncate_mode=truncate_mode,
                p=p_value if truncate_mode else None
            )
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Index")
            ax.set_ylabel("Distance")
            st.pyplot(fig)

        # -----------------------------------------------
        # Display Clustered Data
        # -----------------------------------------------
        st.subheader("Clustered Data Preview")
        st.dataframe(data[['cluster'] + [col for col in data.columns if col != 'cluster']].head())
else:
    st.info("Please upload a dataset or select a sample to begin.")
