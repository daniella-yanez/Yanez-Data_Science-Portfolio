# Yanez-Data-Science-Portfolio

## Purpose  
The purpose of this repository is to showcase my coding projects and demonstrate my proficiency in **data analysis, visualization, machine learning, and research**. Each project highlights my ability to address real-world data challenges using various tools and methodologies. The portfolio emphasizes my skills in cleaning, transforming, analyzing, and visualizing data to extract meaningful insights.

---

## [Tidy Data Project🔗](https://github.com/daniella-yanez/Yanez-Data_Science-Portfolio/tree/7a3ab095e865df407cda39b79a3d1b92b8b242d7/TidyData-Project)

### **Project Overview**  
This project focuses on cleaning and reshaping Olympic medalist data from the 2008 Summer Olympics. The dataset, initially in an untidy format, was transformed into a **tidy format** using **Pandas** and visualized with **Seaborn** and **Matplotlib**. Key tasks involved:
- **Reshaping the data**: Using `pandas.melt()` to convert columns into variables for better analysis.
- **Data cleaning**: Removing duplicates and handling missing values.
- **Categorization**: Converting medals into an ordered categorical variable for easier analysis.

### **Key Techniques and Tools Used**  
- **Pandas** for data manipulation  
- **Seaborn & Matplotlib** for data visualization  
- **Data Transformation** using `.melt()`  
- **Categorization** of medal types  

### **How This Project Complements My Portfolio**  
- **Data Cleaning**: Demonstrates the ability to clean, structure, and transform raw data into a usable format.
- **Visualization**: Shows proficiency in creating insightful visualizations to explore Olympic medal distributions.
- **Tidy Data Principles**: Reinforces the concept of tidy data for reproducible and scalable analysis.

---

## [Machine Learning App – Streamlit Deployment🔗](https://github.com/daniella-yanez/Yanez-Data_Science-Portfolio/tree/7a3ab095e865df407cda39b79a3d1b92b8b242d7/MLStreamlitApp)

### **Project Overview**  
This project features an interactive **Streamlit** web application that allows users to upload datasets, choose supervised machine learning models (Logistic Regression, Decision Tree, or K-Nearest Neighbors), train them, and view performance metrics like accuracy, classification reports, confusion matrices, and ROC curves. The app allows real-time model tuning by adjusting hyperparameters and observing the impact on results.

### **Key Techniques and Tools Used**  
- **Streamlit** for app deployment  
- **Scikit-learn** for machine learning model selection, training, and evaluation  
- **Pandas & Matplotlib** for data handling and visualization  
- **Hyperparameter Tuning** for real-time model adjustment  

### **How This Project Complements My Portfolio**  
- **Python Proficiency**: Highlights the use of key Python libraries like **scikit-learn**, **Pandas**, and **Streamlit**.
- **Machine Learning**: Demonstrates end-to-end implementation of machine learning workflows from data loading to model evaluation.
- **Web Deployment**: First full deployment of a Python app, allowing the interactive exploration of machine learning concepts via Streamlit.
- **User-Focused Design**: Prioritizes ease of use with clear interfaces, making complex machine learning workflows accessible to all users.

---

## [Unsupervised Learning Explorer – Clustering & Dimensionality Reduction Web App🔗](https://github.com/daniella-yanez/Yanez-Data_Science-Portfolio/tree/7a3ab095e865df407cda39b79a3d1b92b8b242d7/MLUnsupervisedApp)

### **Project Overview**  
This Streamlit web application allows users to explore unsupervised learning techniques, specifically **K-Means Clustering**, **Hierarchical Clustering**, and **Principal Component Analysis (PCA)**. The app provides an interactive tool for analyzing high-dimensional datasets, uncovering hidden structures through clustering and dimensionality reduction.

With this app, users can:
- Upload custom datasets (CSV format)
- Select machine learning methods for clustering and dimensionality reduction
- Adjust hyperparameters through interactive widgets
- Visualize results in real-time with intuitive plots
- Evaluate clustering quality with silhouette scores or dendrograms

### **Key Techniques and Tools Used**  
- **Streamlit** for app deployment  
- **Scikit-learn** for implementing K-Means, Hierarchical Clustering, and PCA  
- **Matplotlib & Plotly** for interactive and static visualizations  
- **Silhouette Score** for evaluating clustering quality  
- **Hierarchical Dendrogram** for visualizing cluster relationships  

### **How This Project Complements My Portfolio**  
- **Unsupervised Learning**: Showcases my understanding of clustering and dimensionality reduction techniques, helping to uncover patterns in complex data without labeled examples.
- **Interactivity**: Demonstrates the ability to create user-friendly, interactive applications that allow real-time adjustments and visualizations.
- **Data Exploration**: Provides a tool for non-experts to explore and experiment with machine learning algorithms, making complex techniques more accessible.
- **Web Deployment**: Expands my experience in deploying interactive web applications with **Streamlit**, integrating real-time data processing and model interaction.

---

## How to Use  
Each project repository includes detailed instructions on how to replicate the work, from setting up the environment to running the code. Follow the README in each project directory to get started.

### **To Run Tidy Data Project**  
1. Clone the repository:  
   ```bash  
   git clone <TidyDataProjectRepositoryLink>  
   cd TidyData-Project
2. Open the Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
3. Run the notebook to clean the data and explore visualizations.

### **To Run Machine Learning App**
1. Clone the repository:
  ```bash
  git clone <MLAppRepositoryLink>  
  cd MLStreamlitApp  
  ```
2. Install required dependencies:
  ```bash
  pip install -r requirements.txt  
  ```
3. Start the Streamlit app:
  ```bash
  streamlit run app.py
  ```  
4. Interact with the app in your browser to upload datasets and explore machine learning models.

### **To Run Machine Learning-Unsupervised App**
1. Clone this repo
   ```bash  
   git clone <UnsupervisedLearningExplorerRepositoryLink>  
   cd UnsupervisedLearningExplorer
   ```
2. Install required packages
   ```bash
   pip install -r requirements.txt  

   ```
3. Run the Streamlit App
   ```bash
   streamlit run app.py
   ```
 
4. Interact with the app in your browser to explore the clustering and dimensionality reduction techniques.

---
## Conclusion
This portfolio showcases my ability to handle different types of data science challenges, from cleaning raw datasets to building interactive machine learning applications. By combining my knowledge of Python, data analysis, and web deployment, I aim to solve real-world problems and present data-driven insights in an accessible and impactful way.
