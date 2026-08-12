
# ============================================================
# TASK 1: DATA CLEANING & PREPROCESSING
# Heart Disease Dataset
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. IMPORT DATASET
# ------------------------------------------------------------

df = pd.read_csv("heart.csv")

print("Dataset imported successfully!")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 2. INSPECT DATASET STRUCTURE
# ------------------------------------------------------------

print("\n================ DATASET INFORMATION ================")

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n================ MISSING VALUES ================")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:")
print(df.isnull().sum().sum())


# ------------------------------------------------------------
# 4. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n================ DUPLICATES ================")

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    print("\nDuplicate rows:")
    print(df[df.duplicated()])
else:
    print("No duplicate rows found.")


# ------------------------------------------------------------
# 5. CHECK UNIQUE VALUES / INCONSISTENT ENTRIES
# ------------------------------------------------------------

print("\n================ UNIQUE VALUES ================")

for column in df.columns:
    print("\n", column)
    print(df[column].unique())


# ------------------------------------------------------------
# 6. REMOVE EXTRA SPACES FROM COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

print("\nColumn names after removing spaces:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 7. REMOVE EXTRA SPACES FROM TEXT VALUES
# ------------------------------------------------------------

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()


# ------------------------------------------------------------
# 8. CHECK FOR COMMON INVALID VALUES
# ------------------------------------------------------------

print("\n================ INVALID VALUES ================")

# Replace common representations of missing values
df = df.replace(
    ["?", "NA", "N/A", "na", "null", "NULL", "", " "],
    np.nan
)

print("Missing values after checking invalid entries:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 9. CONVERT COLUMNS TO NUMERIC DATA TYPES
# ------------------------------------------------------------

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
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ------------------------------------------------------------
# 10. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n================ HANDLING MISSING VALUES ================")

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing numeric values using median
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 11. REMOVE DUPLICATE ROWS
# ------------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\n================ DUPLICATE REMOVAL ================")

print("Rows before removing duplicates:", before_duplicates)
print("Rows after removing duplicates:", after_duplicates)
print("Duplicates removed:", before_duplicates - after_duplicates)


# ------------------------------------------------------------
# 12. CHECK DATA CONSISTENCY
# ------------------------------------------------------------

print("\n================ DATA CONSISTENCY CHECK ================")

# Check expected ranges

print("\nAge range:")
print(df["age"].min(), "to", df["age"].max())

print("\nSex values:")
print(df["sex"].unique())

print("\nChest pain (cp) values:")
print(df["cp"].unique())

print("\nFasting blood sugar (fbs) values:")
print(df["fbs"].unique())

print("\nTarget values:")
print(df["target"].unique())


# ------------------------------------------------------------
# 13. REMOVE INVALID VALUES
# ------------------------------------------------------------

# Keep only valid binary values for sex
df = df[df["sex"].isin([0, 1])]

# cp should normally contain values 0-3
df = df[df["cp"].isin([0, 1, 2, 3])]

# fbs should contain 0 or 1
df = df[df["fbs"].isin([0, 1])]

# exang should contain 0 or 1
df = df[df["exang"].isin([0, 1])]

# target should contain 0 or 1
df = df[df["target"].isin([0, 1])]


# ------------------------------------------------------------
# 14. FINAL DATA TYPE CHECK
# ------------------------------------------------------------

print("\n================ FINAL DATA TYPES ================")

print(df.dtypes)


# ------------------------------------------------------------
# 15. FINAL MISSING VALUE CHECK
# ------------------------------------------------------------

print("\n================ FINAL MISSING VALUES ================")

print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())


# ------------------------------------------------------------
# 16. FINAL DUPLICATE CHECK
# ------------------------------------------------------------

print("\n================ FINAL DUPLICATE CHECK ================")

print("Duplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# 17. FINAL DATASET STRUCTURE
# ------------------------------------------------------------

print("\n================ FINAL DATASET ================")

print("Final shape:", df.shape)

print("\nFirst 10 rows:")
print(df.head(10))


# ------------------------------------------------------------
# 18. SAVE CLEANED DATASET
# ------------------------------------------------------------

output_file = "heart_cleaned.csv"

df.to_csv(output_file, index=False)

print("\n================ SUCCESS ================")

print("Cleaned dataset saved successfully as:", output_file)
