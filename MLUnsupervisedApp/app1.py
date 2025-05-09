# ---------------------------------------------------
# Imports: Essential libraries for data handling, ML, and visualization
# ---------------------------------------------------
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
    # File uploader for user dataset
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    # Option to use built-in sample dataset
    use_sample = st.checkbox("Use Sample Dataset (Country Data)", value=False)
    
    # Informational tooltip about clustering methods
    with st.expander("ℹ️ What are the clustering methods?"):
        st.markdown("""
        **K-Means Clustering** partitions data into `k` groups by minimizing the distance between each point and its cluster center. 
        It's fast and works best with spherical clusters of similar size.

        **Hierarchical Clustering** builds a tree of clusters using either bottom-up (agglomerative) or top-down (divisive) merging. 
        It's useful when you want to see nested groupings of data.

        📈 The number of clusters (`k`) determines how many groups the algorithm will try to form. Choosing more clusters can better fit fine-grained patterns but may overfit.
        """)
        
    # Choose clustering method
    clustering_method = st.selectbox("Choose clustering method:", ["K-Means", "Hierarchical"])
    # Slider to choose number of clusters
    n_clusters = st.slider("Number of clusters (k):", 2, 10, 4)
    
    # Default linkage method for Hierarchical Clustering
    linkage_method = "ward"
    if clustering_method == "Hierarchical":
        # User selects linkage strategy if Hierarchical is chosen
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
    # PCA component slider (for visualization only)            
    n_components = st.slider("# PCA Components for Visualization:", 2, 3, 2)
    with st.expander("ℹ️ What does PCA do and what's the difference between 2D vs 3D?"):
        st.markdown("""
        **PCA (Principal Component Analysis)** reduces high-dimensional data into 2 or 3 principal components that capture most of the variance in the data.

        - **2D PCA** is easier to visualize and interpret.
        - **3D PCA** can show more variance but may be harder to interpret due to perspective distortion.

        This transformation helps simplify the dataset while preserving as much information as possible for plotting and clustering.
        """)
# -----------------------------------------------
# Load and Clean Dataset
# -----------------------------------------------
data = None
if uploaded_file:
    # Load user-uploaded dataset
    data = pd.read_csv(uploaded_file)
elif use_sample:
    # Load sample dataset if selected
    try:
        app_dir = os.path.dirname(__file__) #makes app portable
        file_path = os.path.join(app_dir, "data", "Country-data.csv")
        data = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load bundled sample dataset: {e}")

if data is not None:
    # Drop rows with missing values
    initial_rows = data.shape[0]
    data = data.dropna()
    dropped_rows = initial_rows - data.shape[0]

    if dropped_rows > 0:
        st.warning(f"Dropped {dropped_rows} rows with missing values.")
    # Clean and normalize column names
    data.columns = [col.strip().lower() for col in data.columns]
    # Filter numeric columns for analysis
    numeric_data = data.select_dtypes(include=np.number)

    if numeric_data.shape[1] == 0:
        st.error("No numeric features found in the dataset.")
    else:
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Standardize numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Apply PCA for visualization
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Perform selected clustering method
        if clustering_method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            cluster_labels = model.fit_predict(X_scaled)

        # Evaluate clustering with silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        data['cluster'] = cluster_labels

        with st.expander("ℹ️ What does the PCA Scatterplot show?"):
            st.markdown("""
            The scatterplot shows your data projected into 2 or 3 dimensions using PCA, colored by cluster assignment.

            - Each point represents a sample from your dataset.
            - Points close together are similar in terms of the original features.
            - Coloring reflects which cluster each point belongs to, helping visualize groupings.
            """)
        
        # ---------------------------------------------------
        # PCA Scatterplot
        # ---------------------------------------------------
        st.subheader("PCA Scatter Plot")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, edgecolors='k')
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA Clustering Visualization ({clustering_method})")
        st.pyplot(fig)

        with st.expander("ℹ️ What is the Silhouette Score and Heatmap?"):
            st.markdown("""
            - **Silhouette Score** measures how similar a sample is to its own cluster compared to other clusters. 
              Values range from -1 (wrong cluster) to +1 (well-clustered), with values near 0 indicating overlap.
              
            - **Silhouette Heatmap** visually shows the silhouette scores for each point in each cluster:
                - Taller bars = better cohesion within the cluster.
                - Negative bars = possible misclassification.
                - Helps evaluate the quality of clustering and compare methods like K-Means vs Hierarchical.
            """)
        
        # Show silhouette score
        st.markdown(f"**Silhouette Score:** {silhouette_avg:.2f}")

        # -----------------------------------------------
        # Silhouette Score Heatmap (for model comparison)
        # -----------------------------------------------
        st.subheader("Silhouette Score Heatmap (K-Means vs Hierarchical)")

        silhouette_scores = {}

        if clustering_method == "K-Means":
            # Calculate silhouette scores for different k values (K-Means only
            k_values = list(range(2, 11))
            scores = []
            for k in k_values:
                km = KMeans(n_clusters=k, random_state=42)
                labels = km.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                scores.append(score)
            silhouette_scores['K-Means'] = scores

            # Create heatmap
            df_scores = pd.DataFrame(silhouette_scores, index=k_values)
            fig, ax = plt.subplots()
            sns.heatmap(df_scores.T, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_title("Silhouette Scores for K-Means")
            st.pyplot(fig)

        elif clustering_method == "Hierarchical":
            # Calculate scores for multiple linkage methods
            linkage_methods = ["ward", "single", "complete", "average"]
            k_values = list(range(2, 11))
            for method in linkage_methods:
                scores = []
                for k in k_values:
                    model = AgglomerativeClustering(n_clusters=k, linkage=method)
                    labels = model.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, labels)
                    scores.append(score)
                silhouette_scores[method] = scores

            # Create heatmap for hierarchical clustering
            df_scores = pd.DataFrame(silhouette_scores, index=k_values)
            fig, ax = plt.subplots()
            sns.heatmap(df_scores.T, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_title("Silhouette Scores for Hierarchical Linkages")
            st.pyplot(fig)

        # -----------------------------------------------
        # Choropleth Map Visualization (for any dataset with 'country' column)
        # -----------------------------------------------
        st.subheader("World Map Visualization")

        if "country" in data.columns:
            if PLOTLY_AVAILABLE:
                # Generate choropleth if numeric data is available
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
                        # Create interactive map
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
            # Inform user that a 'country' column is required for map plotting
            st.info("To show a choropleth map, your dataset must include a column named 'country'.")
        # If using the sample dataset and it contains a 'country' column, show the map
        if use_sample and "country" in data.columns:
            st.subheader("World Map Visualization by Cluster")
            if PLOTLY_AVAILABLE:
                # Create a choropleth map using Plotly where countries are colored by cluster
                fig = px.choropleth(data, locations="country", locationmode="country names",
                                    color="cluster", title="Country Clusters")
                st.plotly_chart(fig) # Display the map in Streamlit

                # Add an explanation for the map
                with st.expander("ℹ️ What does the World Map show?"):
                    st.markdown("""
                    The map colors countries based on their cluster assignment.

                    ✅ **It only works when:**
                    - You're using the **sample dataset**, or
                    - Your uploaded dataset includes a column with **country names** (standardized).

                    This visualization helps see how clusters relate geographically, useful for global data analysis.
                    """)

        # -----------------------------------------------
        # Elbow Plot for K-Means
        # -----------------------------------------------
        if clustering_method == "K-Means":
            st.subheader("K-Means Elbow Plot")
            distortions = [] # List to hold distortion (inertia) values
            K_range = range(2, 11) # Range of cluster counts to test
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(X_scaled)
                distortions.append(km.inertia_) # Store the distortion (sum of squared distances)
            # Plot the Elbow chart using matplotlib
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
            # Perform hierarchical/agglomerative clustering
            Z = linkage(X_scaled, method=linkage_method)

            # Basic dendrogram plot
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, labels=data.index.to_numpy(), ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Index")
            ax.set_ylabel("Distance")
            st.pyplot(fig)
        
        if clustering_method == "Hierarchical":
            st.subheader("Hierarchical Dendrogram")
            
            # More detailed dendrogram with cutoff line for selected clusters
            linked = linkage(X_scaled, method=linkage_method)
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, truncate_mode="lastp", p=n_clusters, ax=ax, show_contracted=True)
            # Draw red dashed line to represent cluster cut
            ax.axhline(y=linked[-n_clusters, 2], color='r', linestyle='--')
            st.pyplot(fig)

           # User guidance on how to read the dendrogram
            with st.expander("ℹ️ What does the Hierarchical Dendrogram show?"):
                st.markdown("""
                The dendrogram displays how points are merged together into clusters at different distance thresholds.

                - Each branch shows a merge between clusters or points.
                - The **height of the branches** reflects the distance (or dissimilarity) between clusters.
                - Dotted line (if shown) indicates the cut point based on your selected number of clusters.

                ⚠️ **Note**: When using a dataset with many rows or non-unique indexes, the labels may overlap or be difficult to read. Focus on the **structure** of merges more than individual labels.
                """)
        # -----------------------------------------------
        # Display Clustered Data
        # -----------------------------------------------
        st.subheader("Clustered Data Preview")
        # Rearrange columns to show 'cluster' first, then all other columns
        st.dataframe(data[['cluster'] + [col for col in data.columns if col != 'cluster']].head())
        # Explanation comparing original vs clustered data
        with st.expander("📑 What is the difference between the data previews?"):
            st.markdown("""
            - **Initial Dataset Preview** shows your raw uploaded or sample dataset.
            - **Clustered Data Preview** is the same dataset but with an added `Cluster` column, showing the result of your chosen clustering algorithm.

            You can use this column to analyze differences between clusters in your dataset.
            """)

        st.subheader("Clustered Data Preview")
        st.dataframe(data)
else:
    # If no data is available, prompt the user to upload or select a sample
    st.info("Please upload a dataset or select a sample to begin.")


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
