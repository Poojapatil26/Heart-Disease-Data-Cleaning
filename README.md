# ❤️ Heart Disease Data Cleaning & Preprocessing

## 📌 Task 1: Data Cleaning & Preprocessing

This project focuses on **importing, inspecting, cleaning, and preparing a Heart Disease dataset** using Python and Pandas.

The main objective is to identify and handle common data quality issues such as **missing values, duplicate records, inconsistent data entries, and incorrect data types**, and then prepare the cleaned dataset for further data analysis and machine learning.

---

## 🎯 Objectives

The objectives of this task are:

* Import the Heart Disease dataset using Python.
* Inspect the structure and characteristics of the dataset.
* Identify missing/null values.
* Identify duplicate records.
* Detect inconsistent or invalid data entries.
* Handle missing values appropriately.
* Remove duplicate records.
* Correct incorrect data types.
* Validate the cleaned dataset.
* Save the cleaned dataset as a new CSV file.

---

## 📂 Dataset

The dataset used in this project is the **Heart Disease Dataset**.

It contains information related to patients and various health-related attributes that can be used to analyze the presence of heart disease.

### Dataset Columns

| Column     | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Gender of the patient                 |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol level               |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting electrocardiographic results  |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia                           |
| `target`   | Heart disease target variable         |

### Target Variable

The `target` column represents whether heart disease is present.

* `0` → No heart disease
* `1` → Heart disease

---

## 🛠️ Technologies Used

The following tools and technologies were used:

* **Python**
* **Pandas**
* **NumPy**
* **Google Colab**
* **GitHub**
* **CSV**

---

## 📁 Project Structure

```text

Heart-Disease-Data-Cleaning/
│
├── data/
│   └── heart.csv
│
├── heart_data_cleaning.py
│
└── README.md
```

---

# 🔄 Data Cleaning Workflow

## 1. Import the Dataset

The dataset is imported using Pandas.

```python
import pandas as pd

df = pd.read_csv("heart.csv")
```

The first few records are displayed using:

```python
print(df.head())
```

This helps verify that the dataset has been loaded correctly.

---

## 2. Inspect Dataset Structure

The structure of the dataset is inspected using:

```python
print(df.shape)
print(df.columns)
print(df.info())
```

The `shape` function provides the number of rows and columns.

The `columns` function displays all column names.

The `info()` function provides information about:

* Number of records
* Column names
* Data types
* Non-null values

---

## 3. Generate Statistical Summary

The `describe()` function is used to understand the numerical characteristics of the dataset.

```python
print(df.describe())
```

It provides:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

This helps identify unusual or potentially inconsistent values.

---

# 🔍 4. Identify Missing Values

Missing values are checked using:

```python
df.isnull().sum()
```

The total number of missing values can be calculated using:

```python
df.isnull().sum().sum()
```

Missing values can cause problems during analysis and machine learning, so they need to be handled appropriately.

---

# 🔎 5. Identify Duplicate Records

Duplicate rows are identified using:

```python
df.duplicated().sum()
```

If duplicate records are present, they can be displayed using:

```python
df[df.duplicated()]
```

Duplicate records can affect statistical analysis and should generally be removed.

---

# 🔎 6. Check Inconsistent Data Entries

Unique values in each column are inspected using:

```python
for column in df.columns:
    print(column)
    print(df[column].unique())
```

This helps identify unexpected values or inconsistent entries.

For example, categorical columns may contain values outside their expected range.

---

# 🧹 7. Handle Invalid Missing Value Representations

Sometimes missing values are represented using symbols such as:

* `?`
* `NA`
* `N/A`
* `null`
* Empty strings

These values are converted to proper Pandas `NaN` values.

```python
import numpy as np

df = df.replace(
    ["?", "NA", "N/A", "na", "null", "NULL", "", " "],
    np.nan
)
```

After conversion, missing values can be identified using:

```python
df.isnull().sum()
```

---

# 🔧 8. Correct Data Types

The numerical columns are converted into appropriate numeric data types.

```python
numeric_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )
```

Using `errors="coerce"` converts invalid numerical entries into `NaN`, which can then be handled during the cleaning process.

---

# 🩹 9. Handle Missing Values

For numerical columns, missing values are replaced using the **median**.

```python
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].median()
        )
```

### Why Median?

The median is useful for numerical health-related data because it is less affected by extreme values compared with the mean.

---

# 🗑️ 10. Remove Duplicate Records

Duplicate records are removed using:

```python
df = df.drop_duplicates()
```

The number of records before and after removal can be compared to determine how many duplicate records were removed.

---

# ✅ 11. Validate Data Consistency

Important categorical columns are checked to make sure they contain valid values.

For example:

```python
print(df["sex"].unique())
print(df["cp"].unique())
print(df["fbs"].unique())
print(df["target"].unique())
```

Expected values include:

### Sex

```text
0, 1
```

### Chest Pain Type

```text
0, 1, 2, 3
```

### Fasting Blood Sugar

```text
0, 1
```

### Target

```text
0, 1
```

Invalid entries can be removed using conditions such as:

```python
df = df[df["sex"].isin([0, 1])]
df = df[df["cp"].isin([0, 1, 2, 3])]
df = df[df["fbs"].isin([0, 1])]
df = df[df["exang"].isin([0, 1])]
df = df[df["target"].isin([0, 1])]
```

---

# 🔬 12. Final Validation

After cleaning, the dataset is checked again.

### Check missing values

```python
print(df.isnull().sum())
```

### Check duplicates

```python
print(df.duplicated().sum())
```

### Check data types

```python
print(df.dtypes)
```

### Check final shape

```python
print(df.shape)
```

### Display cleaned dataset

```python
print(df.head())
```

The goal is to ensure that the final dataset is clean and ready for further analysis.

---

# 💾 13. Save Cleaned Dataset

The cleaned dataset is saved as a new CSV file.

```python
df.to_csv(
    "heart_cleaned.csv",
    index=False
)
```

The cleaned file is stored separately so that the original dataset remains unchanged.

---

# 📊 Before and After Cleaning

| Data Quality Check | Before Cleaning  | After Cleaning      |
| ------------------ | ---------------- | ------------------- |
| Missing Values     | Checked          | Handled             |
| Duplicate Records  | Identified       | Removed             |
| Invalid Entries    | Identified       | Corrected/Removed   |
| Data Types         | Inspected        | Corrected           |
| Dataset Structure  | Inspected        | Validated           |
| Output             | Original Dataset | `heart_cleaned.csv` |

---

# 🎯 Final Outcome

After completing this task, the Heart Disease dataset is:

* Free from duplicate records.
* Checked for missing values.
* Checked for inconsistent entries.
* Converted to appropriate data types.
* Validated for important categorical values.
* Prepared for further analysis.
* Saved as a separate cleaned CSV file.

The final cleaned dataset is:

```text
heart_cleaned.csv
```

---

# 🚀 Future Analysis

The cleaned dataset can be used for further data analysis and machine learning tasks such as:

* Exploratory Data Analysis (EDA)
* Correlation analysis
* Data visualization
* Feature analysis
* Classification
* Machine learning model development
* Heart disease prediction

---

# 👩‍💻 Author

**Pooja Patil**

M.Tech — Data Science
COEP Technological University, Pune

---

## ⭐ Conclusion

This task demonstrates the complete basic **data cleaning and preprocessing workflow using Python and Pandas**. Proper data cleaning is an important first step before performing exploratory data analysis or building machine learning models because it improves data quality, consistency, and reliability.
