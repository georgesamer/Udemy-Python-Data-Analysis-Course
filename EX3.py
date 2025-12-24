# Import libraries
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

def load_and_prepare_data():
    """Load and prepare CO2 dataset with proper cleaning."""
    # Load dataset
    co2 = sm.datasets.co2.load_pandas().data
    
    # Reset index and rename
    co2 = co2.reset_index().rename(columns={"index": "date"})
    
    # Extract date components more efficiently
    co2["month"] = co2["date"].dt.month
    co2["year"] = co2["date"].dt.year
    co2["month_name"] = co2["date"].dt.strftime("%b")
    
    # Handle missing values
    missing_count = co2["co2"].isna().sum()
    print(f"Missing values found: {missing_count}")
    
    if missing_count > 0:
        co2["co2"] = co2["co2"].interpolate(
            method="polynomial", 
            order=3, 
            limit_direction="both"
        )
        print(f"Interpolated {missing_count} missing values")
    
    return co2

def plot_co2_trends(co2):
    """Create improved line plot showing CO2 trends over time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot with seasonal coloring
    scatter = ax.scatter(
        co2["date"], 
        co2["co2"], 
        c=co2["month"], 
        cmap="coolwarm",
        alpha=0.6,
        s=10
    )
    
    # Add trend line
    z = np.polyfit(range(len(co2)), co2["co2"], 2)
    p = np.poly1d(z)
    ax.plot(co2["date"], p(range(len(co2))), 
            "r--", linewidth=2, alpha=0.8, label="Trend")
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("CO2 Concentration (ppm)", fontsize=12, fontweight="bold")
    ax.set_title("Atmospheric CO2 Levels Over Time", 
                 fontsize=14, fontweight="bold", pad=20)
    
    # Add colorbar for months
    cbar = plt.colorbar(scatter, ax=ax, label="Month")
    cbar.set_ticks(range(1, 13))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_monthly_average(co2):
    """Create improved bar plot showing average CO2 by month."""
    # Calculate monthly statistics
    monthly_stats = co2.groupby("month")["co2"].agg(["mean", "std"]).reset_index()
    monthly_stats["month_name"] = pd.to_datetime(
        monthly_stats["month"], format="%m"
    ).dt.strftime("%b")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bar plot with error bars
    bars = ax.bar(
        monthly_stats["month_name"], 
        monthly_stats["mean"],
        color=sns.color_palette("viridis", 12),
        edgecolor="black",
        linewidth=1.2
    )
    
    # Add error bars
    ax.errorbar(
        monthly_stats["month_name"], 
        monthly_stats["mean"],
        yerr=monthly_stats["std"],
        fmt="none",
        ecolor="black",
        capsize=5,
        alpha=0.7
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.1f}',
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight="bold"
        )
    
    # Formatting
    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average CO2 Concentration (ppm)", fontsize=12, fontweight="bold")
    ax.set_title("Average CO2 Levels by Month (with Standard Deviation)", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig

def print_summary_statistics(co2):
    """Print summary statistics for the dataset."""
    print("\n" + "="*50)
    print("CO2 DATASET SUMMARY STATISTICS")
    print("="*50)
    print(f"\nDate range: {co2['date'].min()} to {co2['date'].max()}")
    print(f"Total observations: {len(co2)}")
    print(f"\nCO2 Concentration Statistics:")
    print(f"  Mean: {co2['co2'].mean():.2f} ppm")
    print(f"  Median: {co2['co2'].median():.2f} ppm")
    print(f"  Std Dev: {co2['co2'].std():.2f} ppm")
    print(f"  Min: {co2['co2'].min():.2f} ppm")
    print(f"  Max: {co2['co2'].max():.2f} ppm")
    print(f"  Range: {co2['co2'].max() - co2['co2'].min():.2f} ppm")
    print("="*50 + "\n")

def main():
    """Main execution function."""
    # Load and prepare data
    co2 = load_and_prepare_data()
    
    # Print summary statistics
    print_summary_statistics(co2)
    
    # Create visualizations
    print("Generating visualizations...")
    plot_co2_trends(co2)
    plot_monthly_average(co2)
    plt.show()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()