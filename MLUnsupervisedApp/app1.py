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
st.title("Machine Learning Clustering App--Unsupervised App")

# -----------------------------------------------
# Sidebar: Data Upload and User Inputs
# -----------------------------------------------
with st.sidebar:
    st.header("Upload and Select Options")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    use_sample = st.checkbox("Use Sample Dataset (Country Data)", value=False)

with st.expander("ℹ️ What are the clustering methods?"):
    st.markdown("""
    **K-Means Clustering** partitions data into `k` groups by minimizing the distance between each point and its cluster center. 
    It's fast and works best with spherical clusters of similar size.

    **Hierarchical Clustering** builds a tree of clusters using either bottom-up (agglomerative) or top-down (divisive) merging. 
    It's useful when you want to see nested groupings of data.

    📈 The number of clusters (`k`) determines how many groups the algorithm will try to form. Choosing more clusters can better fit fine-grained patterns but may overfit.
    """)
  

    clustering_method = st.selectbox("Choose clustering method:", ["K-Means", "Hierarchical"])
    n_clusters = st.slider("Number of clusters (k):", 2, 10, 4)

    linkage_method = "ward"
    if clustering_method == "Hierarchical":
        linkage_method = st.selectbox("Linkage method:", ["ward", "single", "complete", "average"])
        with st.expander("ℹ️ Linkage Methods Explained"):
            st.markdown("""
            Linkage methods determine how distances between clusters are calculated in Hierarchical Clustering:
    
            - **Ward**: Minimizes the variance within clusters; often gives compact and well-separated clusters.
            - **Single**: Uses the shortest distance between any two points in two clusters (can lead to long, chain-like clusters).
            - **Complete**: Uses the furthest distance between points in two clusters (tends to create compact, tight clusters).
            - **Average**: Averages all pairwise distances between points in the two clusters.
    
            Different linkage choices can result in very different dendrogram shapes and cluster assignments.
            """)


    n_components = st.slider("# PCA Components for Visualization:", 2, 3, 2)

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------
data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    try:
        app_dir = os.path.dirname(__file__) #makes app portable
        file_path = os.path.join(app_dir, "data", "Country-data.csv")
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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        if clustering_method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            cluster_labels = model.fit_predict(X_scaled)

        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        data['cluster'] = cluster_labels

        st.subheader("PCA Scatter Plot")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, edgecolors='k')
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA Clustering Visualization ({clustering_method})")
        st.pyplot(fig)

        st.markdown(f"**Silhouette Score:** {silhouette_avg:.2f}")

        # -----------------------------------------------
        # Silhouette Score Heatmap (for model comparison)
        # -----------------------------------------------
        st.subheader("Silhouette Score Heatmap (K-Means vs Hierarchical)")

        silhouette_scores = {}

        if clustering_method == "K-Means":
            k_values = list(range(2, 11))
            scores = []
            for k in k_values:
                km = KMeans(n_clusters=k, random_state=42)
                labels = km.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                scores.append(score)
            silhouette_scores['K-Means'] = scores

            df_scores = pd.DataFrame(silhouette_scores, index=k_values)
            fig, ax = plt.subplots()
            sns.heatmap(df_scores.T, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_title("Silhouette Scores for K-Means")
            st.pyplot(fig)

        elif clustering_method == "Hierarchical":
            linkage_methods = ["ward", "single", "complete", "average"]
            k_values = list(range(2, 11))
            for method in linkage_methods:
                scores = []
                for k in k_values:
                    try:
                        hc = AgglomerativeClustering(n_clusters=k, linkage=method)
                        labels = hc.fit_predict(X_scaled)
                        score = silhouette_score(X_scaled, labels)
                        scores.append(score)
                    except Exception as e:
                        scores.append(np.nan)
                silhouette_scores[method] = scores

            df_scores = pd.DataFrame(silhouette_scores, index=k_values).T
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(df_scores, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
            ax.set_ylabel("Linkage Method")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_title("Silhouette Scores for Hierarchical Clustering")
            st.pyplot(fig)

        # -----------------------------------------------
        # World Map Visualization (for any dataset with 'country' column)
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
        st.dataframe(data[['cluster'] + [col for col in data.columns if col != 'cluster']].head())
else:
    st.info("Please upload a dataset or select a sample to begin.")



# -----------------------------------------------
# Interpret Results
# -----------------------------------------------
st.markdown("### Results Interpretation")
st.write(f"Your model predicts correctly about **{accuracy * 100:.2f}%** of the time.")
st.markdown("- The confusion matrix shows how many predictions were correct (diagonal) vs incorrect (off-diagonal)steru.")
if len(np.unique(y)) == 2:
    st.markdown("- Since this is a binary classification problem, we also show the ROC curve and AUC score as additional metrics.")
else:
    st.markdown("- This is a multiclass classification problem; precision, recall, and F1-score per class help interpret the results.")

# -----------------------------------------------
# Final Note
# -----------------------------------------------
st.markdown("---")
st.markdown("**Next Steps:**")
st.markdown("- Try adjusting the hyperparameters to see how they impact performance.")
st.markdown("- Experiment with other datasets or upload your own to test various models.")

# -----------------------------------------------
# Footer and GitHub Link
# -----------------------------------------------
st.markdown("---")
st.markdown("**For any questions, email daniellamartinezyanez7@gmail.com**")
