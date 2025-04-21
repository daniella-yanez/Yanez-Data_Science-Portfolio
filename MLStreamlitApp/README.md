# Machine Learning Playground: Train Your Own Model with Streamlit


## Project Overview and Purpose
This web app was built with Streamlit to make supervised machine learning accessible, visual, and intuitive — even if you don’t know how to code. Whether you're a beginner exploring ML concepts or just want a fast way to prototype models, this tool lets you:

You can upload your own CSV file or use classic datasets provided, then experiment with different models and hyperparameters to see how they perform.

- Upload your own dataset *or* choose from classic built-in ones like **Option 1**, **Option 2**, or **option 3**
- Train models like **Logistic Regression**, **Decision Tree**, or **K-Nearest Neighbors**
- Tune hyperparameters interactively with sliders and dropdowns
- Visualize model performance with metrics, confusion matrices, and ROC curves

---

## Features

- ✅ Upload your own CSV or try out classic classification datasets  
- ✅ Choose from 3 supervised ML models  
- ✅ Tune hyperparameters directly in the sidebar  
- ✅ View model metrics: Accuracy, Precision, Recall, F1 Score  
- ✅ Interactive charts: Confusion Matrix, ROC Curve, and more  
- ✅ Clean layout using Streamlit expanders and tabs for easy navigation  

---

## How to Use the App
### 🔗 Deploy App
👉 [Check out the live app here!](https://daniella-yanez-hynqnvufsbjcmb8e9ewipq.streamlit.app/)

---

### Run Locally
**Clone this repo**
```bash```
git clone https://github.com/daniella-yanez/MLStreamlitapp.git
cd MLStreamlitapp

## Models and Hyperparameters

* Model * | Key Hyperparameters
* Logistic Regression * | C, penalty, solver
* Decision Tree * | max_depth, criterion, min_samples_split
* KNN * | n_neighbors, weights, metric

## Visuals
### [1. Interface Example](ML_Playground_Interface.png)
This screenshot shows the clean, interactive interface of the Machine Learning Explorer app. It includes:
- A preview of the uploaded dataset
- Automatically detected column types
- An overview of the target variable's distribution
  
### [Dataset Overview Example](Data_Overview_Example.png)
This screenshot displays the top of the interface, showing:
- The uploaded dataset (Iris dataset in this example)
- A dynamic preview of feature columns and the target variable  

### [Results Interpretation Example](Results_Interpretation_Example.png)
This section explains what the model results mean in plain language:
- Summary of model accuracy
- Bullet points interpreting the confusion matrix and classification report
- Suggested next steps for users to refine their model  

## References
[- An Introduction to Statistical Learning ]([url](https://www.statlearning.com/))
[- RMSE vs. R-Squared: Which Metric Should You Use?]([url](https://www.statology.org/rmse-vs-r-squared/))
