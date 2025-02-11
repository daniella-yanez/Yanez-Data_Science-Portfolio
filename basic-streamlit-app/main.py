import streamlit as st
import pandas as pd

# ================================
# Displaying a Simple DataFrame in Streamlit
# ================================

st.subheader("Now, let's look at some penguin data!")

# Load the penguins dataset
#@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url)
    return df

data = load_data()

# Displaying the table in Streamlit
st.write("Here's a sample of the dataset:")
st.dataframe(data.head())

# ================================
# Adding User Interaction with Widgets
# ================================

# Using a selectbox to allow users to filter data by species
species = st.selectbox("Select a species", data["species"].dropna().unique())

# Filtering the DataFrame based on user selection
filtered_data = data[data["species"] == species]

# Display the filtered results
st.write(f"Penguins of species {species}:")
st.dataframe(filtered_data)

# ================================
# Importing Data Using a Relative Path
# ================================

#RELATIVE PATH
#  #df = pd.read_csv("data/sample_data.csv")  # Ensure the "data" folder exists with the CSV file
###df = pd.read_csv("Yanez-Data_Science-Portfolio/basic-streamlit-app/data/penguins.csv")
#df = pd.read_csv("basic-streamlit-app/data/penguins.csv")
st.write("Loading dataset from a local CSV file:")

try:
    local_data = pd.read_csv("data/penguins.csv")  # Ensure the "data" folder exists with the CSV file
    st.dataframe(data.head())
except FileNotFoundError:
    st.write("Local dataset not found. Please upload the CSV file to the 'data' folder.")

# Allowing filtering by island
island = st.selectbox("Select an island", data["island"].dropna().unique())
filtered_data_by_island = data[data["island"] == island]
st.write(f"Penguins on {island} island:")
st.dataframe(filtered_data_by_island)

# ================================
# Simple Data Visualizations
# ================================

st.subheader("Summary Statistics")
st.write("Basic statistics of the dataset:")
st.write(data.describe())

st.subheader("Feature Comparisons")
st.write("Check how different numerical features compare across species.")
option = st.radio("Select a feature to analyze:", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

# Display averages based on the selected feature
avg_values = data.groupby("species")[option].mean()
st.write(f"Average {option} by species:")
st.dataframe(avg_values)


