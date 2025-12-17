# Tidy Data Project: Restructuring Olympic Medalist Data

## Project Overview and Purpose

This project focuses on restructuring a dataset of **Olympic medalists from the 2008 Olympics** into a **tidy format**, following the **tidy data principles** outlined by Hadley Wickham.

The goals of this project are to:

- **Reshape the dataset** for improved analysis and visualization  
- **Clean and organize the data** by removing duplicates, handling missing values, and categorizing medals  
- **Explore the data** through visualizations using Seaborn and Matplotlib  

---

## Dataset Description

- **Source**: The dataset contains information on Olympic medalists from 2008, including name, sport, gender, and medal type.

- **Preprocessing Steps**:  
  - **Data Cleaning**: Removed duplicates and missing values  
  - **Data Transformation**: Used `.melt()` to reshape the data  
  - **Feature Engineering**: Extracted gender and sport from a combined column  
  - **Categorization**: Converted "Medal" into an ordered categorical variable  

---

## How to Run the Project

### Run Locally

**Clone this repo:**
```bash
git clone <repository_link>
cd TidyData-Project
```

Install required packages:

```bash
pip install pandas matplotlib seaborn jupyter
```
Open the notebook:
```bash
jupyter notebook
Run TidyData-Project.ipynb to execute the data cleaning and visualization steps.
```
## Visual Examples

### 1. Untidy Dataset: Common Issues

- The original dataset contains **70+ columns** for men’s and women’s sports, creating an unreadable structure.  
- Medals (**Gold**, **Silver**, **Bronze**) are spread across too many columns.  
- Answering simple questions about a specific medalist, sport, or medal type is incredibly difficult.  
- Below: **30 rows and 4 columns** capture only **3 medals** — an example of this chaotic structure.

<img width="916" alt="Screenshot 2025-03-18 at 7 25 28 AM" src="https://github.com/user-attachments/assets/55b6659c-d2ca-463b-8dbc-f820a00d4011" />

---

### 2. Improved, Tidy Dataset

- Cleaned and reshaped version using `pandas.melt()`  
- Structured for easy analysis and visualization

<img width="350" alt="Screenshot 2025-03-18 at 7 28 48 AM" src="https://github.com/user-attachments/assets/bc9bf35f-8aec-4b99-a25d-3e97aa9fac0c" />

---

## References 
- [Tidy Data Paper by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

---

## Contact
**For any questions, email daniellamartinezyanez7@gmail.com**
