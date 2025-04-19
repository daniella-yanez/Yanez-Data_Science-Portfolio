import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.utils import shuffle

# -----------------------------------------------
# Load Sample Datasets
# -----------------------------------------------
def get_sample_dataset(name):
    if name == 'Iris':
        data = load_iris()
    elif name == 'Wine':
        data = load_wine()
    elif name == 'Breast Cancer':
        data = load_breast_cancer()
    elif name == 'Digits':
        data = load_digits()
    else:
        return None, None, None
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names, data.target_names

# -----------------------------------------------
# Sidebar Configuration
# -----------------------------------------------
st.sidebar.title("ML Playground")
dataset_source = st.sidebar.radio("Select Dataset Source", ["Sample Dataset", "Upload Your Own"])
model_type = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

# Hyperparameters
st.sidebar.subheader("Model Hyperparameters")
if model_type == "Logistic Regression":
    C_val = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
elif model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
elif model_type == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("n_neighbors (k)", 1, 15, 5)

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------
if dataset_source == "Sample Dataset":
    dataset_name = st.sidebar.selectbox("Choose Sample Dataset", ["Iris", "Wine", "Breast Cancer", "Digits"])
    df, feature_names, target_names = get_sample_dataset(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        feature_names = df.columns[:-1]
        target_names = df.iloc[:, -1].unique()
    else:
        st.warning("Upload a dataset to continue.")
        st.stop()

# Preview Data
st.title("Interactive ML App")
st.markdown("### Dataset Overview")
st.dataframe(df.head())
st.write("**Class Distribution in Target**")
if 'target' in df.columns:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='target', ax=ax)
    st.pyplot(fig)
    plt.clf()
else:
    st.warning("No 'target' column found in dataset.")
    st.stop()

# -----------------------------------------------
# Train/Test Split & Scaling
# -----------------------------------------------
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------
# Model Training
# -----------------------------------------------
if model_type == "Logistic Regression":
    model = LogisticRegression(C=C_val, max_iter=1000)
elif model_type == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=max_depth)
elif model_type == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------------------------
# Performance Metrics
# -----------------------------------------------
st.markdown("### Model Performance")
st.write(f"**Model**: {model_type}")
st.write(f"**Accuracy**: {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
plt.clf()

# Classification Report
st.subheader("Classification Report")
try:
    st.text(classification_report(y_test, y_pred, target_names=[str(name) for name in target_names]))
except Exception as e:
    st.text(classification_report(y_test, y_pred))

# ROC Curve for Binary Classification
if len(np.unique(y)) == 2:
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.warning("Could not plot ROC curve: " + str(e))

# -----------------------------------------------
# Interpret Results
# -----------------------------------------------
st.markdown("### Results Interpretation")
st.write(f"Your model predicts correctly about **{accuracy * 100:.2f}%** of the time.")
st.markdown("- The confusion matrix shows how many predictions were correct (diagonal) vs incorrect (off-diagonal).")
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
st.markdown("**Made with ❤️ by Emily | [View on GitHub](https://github.com/emily-portfolio/ml-streamlit-app)**")
