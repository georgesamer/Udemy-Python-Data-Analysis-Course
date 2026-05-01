import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = r"D:\VS_code\VS_code_WorkSpace\python_projects\learn\Coffe_sales.csv"
NULL_PLACEHOLDERS = {"unknown", "nan", "none", "n/a", "error", ""}
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_STD  = (10, 6)

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and immediately parse the Date column."""
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df

# ── Cleaning ──────────────────────────────────────────────────────────────────
def clean_column_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Strip / lowercase / underscore-space all string columns.
    - Replace known null-like strings with NaN.
    - Fill remaining NaNs:
        • numeric  → median  (robust to outliers)
        • object   → mode
    """
    df = df.copy()                          # never mutate the caller's frame

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)   # handles multi-spaces
            )
            df[col] = df[col].replace(
                {ph: np.nan for ph in NULL_PLACEHOLDERS}
            )

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df

# ── Plotting helpers ───────────────────────────────────────────────────────────
def _save_or_show(title: str) -> None:
    """Finalise a figure: tight layout, then display."""
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_daily_trend(df: pd.DataFrame) -> None:
    """Line chart: daily units sold per coffee type."""
    daily = (
        df.groupby(["Date", "coffee_name"])
        .size()
        .reset_index(name="sales_count")
    )
    _, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    sns.lineplot(data=daily, x="Date", y="sales_count",
                 hue="coffee_name", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    plt.xticks(rotation=45)
    _save_or_show("Daily Sales Trend per Coffee Type")


def plot_total_revenue(df: pd.DataFrame) -> None:
    """
    Bar chart: TOTAL REVENUE per coffee type.
    Uses estimator=sum so the bar height = Σ money (not a count).
    """
    _, ax = plt.subplots(figsize=FIGURE_SIZE_STD)
    order = (
        df.groupby("coffee_name")["money"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    sns.barplot(data=df, x="coffee_name", y="money",
                estimator="sum", order=order, ax=ax)
    ax.set_xlabel("Coffee Type")
    ax.set_ylabel("Total Revenue")
    plt.xticks(rotation=45)
    _save_or_show("Total Revenue per Coffee Type")


def plot_sales_distribution(df: pd.DataFrame) -> None:
    """
    Count plot: how many transactions per coffee type.
    countplot is the correct chart for a categorical frequency —
    histplot expects a numeric x axis.
    """
    _, ax = plt.subplots(figsize=FIGURE_SIZE_STD)
    order = df["coffee_name"].value_counts().index
    sns.countplot(data=df, x="coffee_name", order=order, ax=ax)
    ax.set_xlabel("Coffee Type")
    ax.set_ylabel("Number of Transactions")
    plt.xticks(rotation=45)
    _save_or_show("Transaction Count per Coffee Type")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_column_data(df)

    plot_daily_trend(df)
    plot_total_revenue(df)
    plot_sales_distribution(df)


if __name__ == "__main__":
    main()