# Import libraries
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

# Import data
df = sm.datasets.get_rdataset('EuroEnergy', 'AER', cache=True).data
df.reset_index(inplace=True)

# Load world data with Path for cross-platform compatibility
file_path = Path("D:/VS_code/VS_code_WorkSpace/python_projects/MTF/EX1_Database/ne_110m_admin_0_countries_lakes.shp")
world = gpd.read_file(file_path)

# Filter and clean world data
world = world[world["CONTINENT"] == "Europe"].copy()  # Use .copy() to avoid SettingWithCopyWarning
world.columns = world.columns.str.lower()

# Data cleaning
df.rename(columns={"rownames": "country"}, inplace=True)
df["country"].replace({"WGermany": "Germany", "UK": "United Kingdom"}, inplace=True)

# Create energy categories with more descriptive bins
bins = [0, 5000, 30000, df["energy"].max()]
labels = ["Low", "Medium", "High"]
df["energy_category"] = pd.cut(df["energy"], bins=bins, labels=labels, include_lowest=True)

# Combine data sources
df = world.merge(df, left_on="name", right_on="country", how="left")
df = df[["name", "energy", "economy", "geometry", "country", "energy_category"]]
df["energy"] = df["energy"].fillna(0)

# Data visualization - Map
fig, ax = plt.subplots(figsize=(12, 10))
df.plot(
    ax=ax, 
    column="energy", 
    legend=True, 
    cmap="YlOrRd",  # More intuitive color scheme for energy
    edgecolor="black",
    legend_kwds={'label': "Energy Consumption", 'orientation': "vertical"}
)
ax.set_xlim(-15, 35)
ax.set_ylim(32, 72)
ax.set_title("Energy Consumption Across European Countries", fontsize=14, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.axis('off')  # Remove axes for cleaner map
plt.tight_layout()
plt.show()

# Barplot - with improvements
fig, ax = plt.subplots(figsize=(10, 6))
# Filter out NaN values for cleaner plot
plot_data = df.dropna(subset=['economy', 'energy'])
sns.barplot(
    data=plot_data, 
    y="economy", 
    x="energy", 
    errorbar=None, 
    estimator="mean",
    palette="viridis",
    ax=ax
)
ax.set_xlabel("Energy Consumption", fontsize=12)
ax.set_ylabel("Economy Type", fontsize=12)
ax.set_title("Average Energy Consumption by Economy Type", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()