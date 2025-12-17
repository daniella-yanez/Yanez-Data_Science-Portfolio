# Penguin Data Streamlit App

## Overview  
This Streamlit app allows users to explore and analyze the Palmer Penguins dataset through interactive widgets and basic data visualizations. The app loads data dynamically from an online source and includes optional functionality for local CSV loading. It provides filtering by species and island, summary statistics, and simple feature comparisons across species.

## Features  
- ✅ Displays the dataset with sample rows for a quick overview  
- ✅ Filters data based on selected **species** using a dropdown widget  
- ✅ Filters data by **island** to allow geographic exploration  
- ✅ Loads data from a remote source (Seaborn GitHub) and attempts to load from a **local CSV** as a backup  
- ✅ Shows **summary statistics** for all numeric features (e.g., mean, std, min, max)  
- ✅Enables **feature comparisons** (bill length, depth, flipper length, body mass) across penguin species via a radio button selector  
- ⚠️ Error handling for missing local files with a user-friendly message  

## How to Run  
### Prerequisites  
Make sure you have Streamlit and Pandas installed:  
```bash  
pip install streamlit pandas  
```

##Steps
Clone the repository:
```
bash
git clone <repository_link>  
cd penguin-data-streamlit-app  
```
Run the Streamlit app:
```
bash
streamlit run app.py
```
Interact with the widgets in your browser to filter and explore the dataset.

## Data Source 🔗
[Palmer Penguins Dataset (via Seaborn)](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)

This project is a basic introduction to building interactive data applications using Streamlit.

---

## Contact
**For any questions, email daniellamartinezyanez7@gmail.com**




