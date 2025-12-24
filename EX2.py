# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_URL = "https://www.energyinst.org/__data/assets/excel_doc/0007/1055545/EI-stats-review-all-data.xlsx"
LOCAL_FILE = Path("D:/VS_code/VS_code_WorkSpace/python_projects/MTF/EI-stats-review-all-data.xlsx")

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class EnergyDataAnalyzer:
    """Handles loading and analysis of energy and emissions data."""
    
    def __init__(self, filepath: Path):
        """Initialize with Excel file path."""
        self.filepath = filepath
        self.xls = None
        self._load_excel()
    
    def _load_excel(self) -> None:
        """Load Excel file and display available sheets."""
        try:
            self.xls = pd.ExcelFile(self.filepath)
            print(f"Available sheets: {', '.join(self.xls.sheet_names)}\n")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.filepath}")
    
    def extract_data(
        self, 
        sheet: str, 
        country: str = "Total World", 
        column_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract and clean data from a specific sheet.
        
        Args:
            sheet: Name of the Excel sheet
            country: Country/region to extract (default: "Total World")
            column_name: Name for the output column (default: uses country name)
        
        Returns:
            DataFrame with cleaned data for the specified country
        """
        # Read and clean data
        df = pd.read_excel(self.filepath, sheet_name=sheet, skiprows=2)
        df.rename(columns={df.columns[0]: "country"}, inplace=True)
        df = df[df.columns[:-3]]  # Remove last 3 columns (typically notes)
        df.set_index("country", inplace=True)
        df.replace({"-": 0, "N/A": 0}, inplace=True)
        
        # Convert to numeric, handling any remaining non-numeric values
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Extract specific country data
        if country not in df.index:
            available = df.index.tolist()[:10]
            raise ValueError(f"Country '{country}' not found. Available: {available}...")
        
        country_data = pd.DataFrame(df.loc[country])
        col_name = column_name or country
        country_data.rename(columns={country: col_name}, inplace=True)
        
        return country_data
    
    def get_full_sheet(self, sheet: str) -> pd.DataFrame:
        """Get complete cleaned data from a sheet."""
        df = pd.read_excel(self.filepath, sheet_name=sheet, skiprows=2)
        df.rename(columns={df.columns[0]: "country"}, inplace=True)
        df = df[df.columns[:-3]]
        df.set_index("country", inplace=True)
        df.replace({"-": 0, "N/A": 0}, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"Missing values in {sheet}: {df.isna().sum().sum()}")
        return df


def plot_emissions_heatmap(data: pd.DataFrame, figsize: tuple = (14, 10)) -> None:
    """Plot heatmap of CO2 emissions across countries and years."""
    plt.figure(figsize=figsize)
    sns.heatmap(data, cmap="crest", cbar_kws={'label': 'CO2e Emissions'})
    plt.title("CO2 Emissions Heatmap by Country and Year", fontsize=16, pad=20)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Country", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame, figsize: tuple = (10, 8)) -> None:
    """Plot correlation matrix between different energy sources and emissions."""
    plt.figure(figsize=figsize)
    correlation = data.corr()
    
    sns.heatmap(
        correlation, 
        cmap="coolwarm", 
        annot=True, 
        fmt='.2f',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title("Correlation: Carbon Emissions vs Energy Sources", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print correlation insights
    print("\nCorrelation with Carbon Emissions:")
    print(correlation['Carbon'].sort_values(ascending=False))


def main():
    """Main analysis workflow."""
    print("=" * 60)
    print("Energy and Carbon Emissions Analysis")
    print("=" * 60 + "\n")
    
    # Initialize analyzer
    analyzer = EnergyDataAnalyzer(LOCAL_FILE)
    
    # Load full carbon emissions data
    print("Loading CO2 emissions data...")
    carbon_full = analyzer.get_full_sheet("CO2e Emissions")
    
    # Extract world-level data for different energy sources
    print("\nExtracting global energy data...")
    data_sources = {
        "Carbon": ("CO2e Emissions", "Carbon"),
        "Solar": ("Solar Generation - TWh", "Solar"),
        "Wind": ("Wind Generation - TWh", "Wind"),
        "Hydro": ("Hydro Generation - TWh", "Hydro"),
        "Nuclear": ("Nuclear Generation - TWh", "Nuclear"),
        "Biomass": ("Geo Biomass Other - TWh", "Biomass")
    }
    
    dataframes = []
    for name, (sheet, col_name) in data_sources.items():
        try:
            df = analyzer.extract_data(sheet, "Total World", col_name)
            dataframes.append(df)
            print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
    
    # Combine all data
    combined_data = pd.concat(dataframes, axis=1)
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(combined_data.describe())
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_emissions_heatmap(carbon_full)
    plot_correlation_matrix(combined_data)
    
    print("\n✓ Analysis complete!")
    
    return combined_data, carbon_full


if __name__ == "__main__":
    combined_data, carbon_full = main()