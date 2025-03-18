#Set-up
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Loading and Inspecting Data
file_path = 'olympics_08_medalists.csv'
df = pd.read_csv(file_path)
print("Below is the beginning of an unclean, horribly messy dataset regarding which medal each medalist won.\n Let's look at the first few rows...")
display(df.head())
print("Let's fix it!")

#Loading and Inspecting Data cont...
print("\nData, Column, etc Dataset Info:")
df.info()

#Loading and Inspecting Data cont...
print("\nBelow are Missing Values in the Dataset:")
print(df.isnull().sum())

###Cleaning and tidying messy dataset
#resharping and organizing data
#id_vars = names
#values = medal type, gender, sport/event
#observations = each individual
df_tidy = (
    df
    .drop_duplicates()
    .melt(id_vars=["medalist_name"], 
        var_name="Event_Gender", 
        value_name="Medal")
    
)
# Remove empty values: 
# this is necessary because of the large gaps in observations since we don't need to specify which of the many sport the medalist didn't get a medal for.
df_tidy = df_tidy.dropna()

df_tidy = df_tidy[df_tidy["Medal"] != ""]
#Removed EMPTY SPACES, particularly for medals since not every medalist received an award for every event

# Make event and gender own variables; Separate Event and Gender instead of doing str.split and str.replace
df_tidy[['Gender', 'Sport']] = df_tidy["Event_Gender"].str.extract(r'(\w+)_(.+)')

# Reorder and clean columns
df_tidy = df_tidy[['medalist_name', 'Sport', 'Gender', 'Medal']].sort_values(by=["Sport", "Medal"])

# Save cleaned dataset
df_tidy.to_csv("tidy_olympics_08_medalists.csv", index=False)

df_tidy.head()
df_tidy = df_tidy.sort_values(by=["Sport", "Medal"])

# Save cleaned dataset
df_tidy.to_csv("tidy_olympics_08_medalists.csv", index=False)

df_tidy.head()



#Variable needs to have its own column
#Variables: Sport/Event, Medal Tyle, Gender
print("Below is a comparison of the components that used to make up the dataset before and after tidying.")
print("Before: ")
df.info()
print("After: ")
df_tidy.info()


#Visualization 1: count plot that demonstrates which sport had the most amount of medalists.
plt.figure(figsize=(12,6)) 
sns.countplot(x='Sport', data=df_tidy, palette='viridis')
plt.xticks(rotation=90)  # Rotate labels
plt.show()


#Visualization 2 shows the concentration of each metal across gender
heatmap_data = df_tidy.pivot_table(index="Gender", columns="Medal", aggfunc="size", fill_value=0)

sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt="d")
plt.title("Medal Distribution by Gender")
plt.show()

#Pivot Table
pivot_table = df_tidy.pivot_table(
    index="Sport",       # Rows (Index)
    columns="Gender",    # Columns (Grouping by Gender)
    values="Medal",      # The value being aggregated
    aggfunc="count",     # Aggregation function (counting medals)
    fill_value=0         # Fill NaNs with 0
)


print("Pivot table of metals obtained by male and female athletes per sport!")
print(pivot_table)
#pivot_table

#Patterns?
print("Descibing our dataset")
df_tidy.describe()

'''
Basic Exploratory Data Analysis
'''
#Since there are a total of 187 medalist names counted and 187 unique medalist names, we can confirm that there are no duplicates
'''
The heat graph shows that there were more bronze medals awarded than silver or gold. 
Also, the heatmap demonstrates that overall, there were more medals distributed to males than females.
'''
