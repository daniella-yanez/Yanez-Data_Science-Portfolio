import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# -----------------------------------------------
#                 Title and Info
# -----------------------------------------------
st.title("Supervised Machine Learning: Model Explorer")

st.markdown("""
Welcome to this interactive classifier app!
This app provides an interactive interface to experiment with different supervised machine learning models. 

In this app, you can:
- **Upload your own dataset** or choose from built-in sample datasets.
- **Adjust hyperparameters** for K-Nearest Neighbors, Logistic Regression, and Decision Tree classifiers.
- **View performance metrics**, including accuracy, precision, recall, ROC curves, and more.
""")

# -----------------------------------------------
#           Load and Preprocess Data
# -----------------------------------------------
# Function to load datasets
def load_data(dataset_option):
    if dataset_option == "Iris":
        data = load_iris()
    elif dataset_option == "Wine":
        data = load_wine()
    elif dataset_option == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_option == "Digits":
        data = load_digits()
    return pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target, name="target")

def upload_custom_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(df.head())
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
        y = df['target']
    else:
        st.error("No target column found. Please make sure your data has a 'target' column.")
        return None, None
    return X, y

# Function to train selected model
def train_model(X_train, y_train, model_name, params):
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["k"])
    elif model_name == "Logistic Regression":
        model = LogisticRegression(C=params["C"], max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=params["max_depth"])
    model.fit(X_train, y_train)
    return model

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy: {accuracy:.2f}**")

    # Classification report
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    # ROC Curve (for binary classification)
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({model_name})')
        plt.legend(loc="lower right")
        st.pyplot(plt)
        plt.clf()

# -----------------------------------------------
#             Streamlit App Layout
# -----------------------------------------------

# Dataset selection
st.markdown("### Select or Upload a Dataset")
dataset_option = st.radio("Choose a dataset", options=["Iris", "Wine", "Breast Cancer", "Digits", "Upload Custom CSV"])

# If custom upload
if dataset_option == "Upload Custom CSV":
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file:
        X, y = upload_custom_data(uploaded_file)
    else:
        st.write("Please upload a file to proceed.")
else:
    X, y = load_data(dataset_option)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter controls
st.markdown("### Select Model and Hyperparameters")
model_name = st.selectbox("Select Model", ["KNN", "Logistic Regression", "Decision Tree"])

if model_name == "KNN":
    k = st.slider("Select number of neighbors (k)", min_value=1, max_value=21, step=2, value=5)
    params = {"k": k}
elif model_name == "Logistic Regression":
    C = st.slider("Select C (regularization parameter)", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
    params = {"C": C}
elif model_name == "Decision Tree":
    max_depth = st.slider("Select max depth", min_value=1, max_value=20, value=5)
    params = {"max_depth": max_depth}

# Model training and evaluation
if st.button("Train Model"):
    # Train selected model
    model = train_model(X_train, y_train, model_name, params)
    # Evaluate model performance
    evaluate_model(model, X_test, y_test, model_name)
    
# -----------------------------------------------
#         Conclusion and Additional Info
# -----------------------------------------------
st.markdown("### Next Steps")
st.write("""
Once you have explored the model, you can:
- Try adjusting the hyperparameters to see how they impact performance.
- Experiment with other datasets or upload your own to test various models.
""")
