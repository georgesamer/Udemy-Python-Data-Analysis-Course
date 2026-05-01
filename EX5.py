import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
FILE_PATH = r"D:\VS_code\VS_code_WorkSpace\python_projects\learn\HR_Analytics.csv"

df = pd.read_csv(FILE_PATH)

# ─────────────────────────────────────────────
# 2. Basic EDA
# ─────────────────────────────────────────────
def eda_summary(df: pd.DataFrame) -> None:
    """Print a structured overview of the dataframe."""
    print(f"Shape: {df.shape}")
    print("=" * 60)

    print("Columns:\n", df.columns.tolist())
    print("=" * 60)

    print("Data Types & Non-Null Counts:")
    df.info()
    print("=" * 60)

    print("Descriptive Statistics:")
    print(df.describe(include="all").T)
    print("=" * 60)

    print("Sample (5 rows):")
    print(df.sample(5, random_state=42))
    print("=" * 60)

    print("Missing Values per Column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"Missing": missing, "% Missing": missing_pct})
            .query("Missing > 0")
            .sort_values("Missing", ascending=False))
    print("=" * 60)


eda_summary(df)

# ─────────────────────────────────────────────
# 3. Column-Level Exploration
# ─────────────────────────────────────────────
CATEGORICAL_COLS = ["SalarySlab", "Department", "Gender", "Attrition"]
NUMERICAL_COLS   = ["Age", "MonthlyIncome", "YearsAtCompany"]  # adjust to your actual columns

for col in CATEGORICAL_COLS:
    if col in df.columns:
        print(f"\n[{col}] value counts:")
        print(df[col].value_counts())

# ─────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")

# --- 4a. Employee count per Department (bar) ---
if "Department" in df.columns:
    plt.figure(figsize=(10, 5))
    order = df["Department"].value_counts().index
    sns.countplot(data=df, x="Department", order=order)
    plt.title("Employee Count by Department")
    plt.xlabel("Department")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

# --- 4b. SalarySlab distribution (count) ---
if "SalarySlab" in df.columns:
    plt.figure(figsize=(10, 5))
    salary_order = df["SalarySlab"].value_counts().index
    sns.countplot(data=df, x="SalarySlab", order=salary_order)
    plt.title("Employee Count by Salary Slab")
    plt.xlabel("Salary Slab")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# --- 4c. SalarySlab × Department (heatmap of counts) ---
if {"SalarySlab", "Department"}.issubset(df.columns):
    pivot = df.groupby(["Department", "SalarySlab"]).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Employee Distribution: Department × Salary Slab")
    plt.tight_layout()
    plt.show()

# --- 4d. Age distribution ---
if "Age" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Age"], bins=20, kde=True, color="steelblue")
    plt.title("Age Distribution of Employees")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.show()

# --- 4e. Attrition rate by SalarySlab ---
if {"Attrition", "SalarySlab"}.issubset(df.columns):
    attrition_rate = (
        df.groupby("SalarySlab")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Attrition Rate (%)")
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=attrition_rate, x="SalarySlab", y="Attrition Rate (%)")
    plt.title("Attrition Rate (%) by Salary Slab")
    plt.xlabel("Salary Slab")
    plt.ylabel("Attrition Rate (%)")
    plt.tight_layout()
    plt.show()