
# Essential library imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# --- Global Configuration ---
# Set plotting style and resolution for better visuals
sns.set_theme(style="ticks", context="notebook")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.autolayout"] = True

# --- Constants ---
RESULTS_DIR = Path("outputs") # Directory to save all artifacts
POSSIBLE_LOCAL_FILES = [
    "Unemployment in India.csv",
    "unemployment.csv",
    "Unemployment_in_India.csv",
    "unemployment-in-india.csv"
]
KAGGLE_SLUG = "gokulrajkmv/unemployment-in-india"
FALLBACK_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/unemployment.csv"

# --- Data Loading and Preparation ---

def get_unemployment_data():
    """
    Attempts to load the unemployment dataset from various sources in order:
    1. Local files in the current directory.
    2. Kaggle API download.
    3. A fallback URL from a public GitHub repository.

    Raises:
        FileNotFoundError: If the dataset cannot be loaded from any source.
    """
    # 1. Try loading from local files
    for filepath in POSSIBLE_LOCAL_FILES:
        if Path(filepath).exists():
            print(f"Found and loading local file: {filepath}")
            return pd.read_csv(filepath)

    # 2. Try downloading via Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        print(f"Attempting to download from Kaggle: {KAGGLE_SLUG}")
        api = KaggleApi()
        api.authenticate()
        
        kaggle_dest = Path("kaggle_data")
        kaggle_dest.mkdir(exist_ok=True)
        api.dataset_download_files(KAGGLE_SLUG, path=kaggle_dest, unzip=True)
        
        csv_files = list(kaggle_dest.glob("*.csv"))
        if csv_files:
            print(f"Successfully downloaded and loading: {csv_files[0]}")
            return pd.read_csv(csv_files[0])
    except Exception as e:
        print(f"Kaggle API download failed. Reason: {e}")

    # 3. Try the fallback URL
    try:
        print(f"Attempting to load from fallback URL: {FALLBACK_URL}")
        return pd.read_csv(FALLBACK_URL)
    except Exception as e:
        print(f"Fallback URL download failed. Reason: {e}")

    # If all methods fail, raise an error
    error_message = textwrap.dedent(f"""
    FATAL: Could not load the dataset. Please ensure you have one of the following:
    1. A local file named one of: {', '.join(POSSIBLE_LOCAL_FILES)}
    2. The 'kaggle' library installed and your API token (kaggle.json) configured correctly.
    3. An active internet connection to access the fallback URL.
    """)
    raise FileNotFoundError(error_message)

def standardize_column_names(df):
    """
    Renames DataFrame columns to a consistent, standardized format.
    Handles various possible original column names for key metrics.
    """
    # Define mappings from standard names to possible original names
    column_map_config = {
        "region": ["Region", "State"],
        "date": ["Date", "Month"],
        "frequency": ["Frequency"],
        "unemployment_rate": ["Estimated Unemployment Rate (%)", "Unemployment Rate"],
        "estimated_employed": ["Estimated Employed"],
        "labour_participation_rate": ["Estimated Labour Participation Rate (%)", "Labour Participation Rate"],
        "area": ["Area"],
        "longitude": ["Longitude"],
        "latitude": ["Latitude"]
    }

    rename_dict = {}
    current_cols = df.columns.tolist()
    
    for standard_name, possible_names in column_map_config.items():
        for name in possible_names:
            if name in current_cols:
                rename_dict[name] = standard_name
                break # Move to the next standard name once a match is found
    
    # Clean up column names with leading/trailing spaces
    df = df.rename(columns=lambda c: c.strip())
    
    # Apply the standardized renaming
    df = df.rename(columns=rename_dict)
    
    print("Standardized columns:", list(df.columns))
    return df

def clean_and_process_data(df):
    """
    Performs all cleaning and feature engineering steps on the raw DataFrame.
    """
    # Standardize column names first
    df = standardize_column_names(df)

    # Clean string columns by stripping whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Convert date column to datetime objects
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # Clean and convert numeric columns
    numeric_cols = ["unemployment_rate", "estimated_employed", "labour_participation_rate", "longitude", "latitude"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any duplicate rows to ensure data integrity
    df = df.drop_duplicates().reset_index(drop=True)

    # Engineer time-based features
    if 'timestamp' in df.columns:
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month_name()
        df['period'] = df['timestamp'].dt.to_period('M').astype(str)
    
    return df

# --- Visualization Functions ---

def plot_national_trend(df, save_dir):
    """Plots the national average unemployment rate over time."""
    if 'period' not in df.columns: return
    
    national_avg = df.groupby('period')['unemployment_rate'].mean().reset_index()
    national_avg['period_dt'] = pd.to_datetime(national_avg['period'])
    
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=national_avg, x='period_dt', y='unemployment_rate', marker='.', color='darkblue')
    plt.title("National Average Unemployment Rate Trend", fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Mean Unemployment Rate (%)")
    plt.savefig(save_dir / "01_national_timeseries.png")
    plt.show()

def plot_top_regions_bar(df, save_dir, n=10):
    """Plots a bar chart of the top N regions with the highest unemployment."""
    if 'region' not in df.columns: return
    
    region_avg = df.groupby('region')['unemployment_rate'].mean().nlargest(n).reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=region_avg, x='unemployment_rate', y='region', palette='viridis')
    plt.title(f"Top {n} Regions by Average Unemployment", fontweight='bold')
    plt.xlabel("Average Unemployment Rate (%)")
    plt.ylabel("Region")
    plt.savefig(save_dir / "02_top_regions_barchart.png")
    plt.show()

def plot_regional_trends(df, save_dir, n=7):
    """Plots time series trends for the N regions with the highest unemployment."""
    if 'region' not in df.columns or 'period' not in df.columns: return

    top_regions_list = df.groupby('region')['unemployment_rate'].mean().nlargest(n).index
    df_top = df[df['region'].isin(top_regions_list)]
    
    df_top['period_dt'] = pd.to_datetime(df_top['period'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_top, x='period_dt', y='unemployment_rate', hue='region', style='region', markers=True)
    plt.title(f"Unemployment Trends for Top {n} Regions", fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(save_dir / "03_top_regions_trends.png")
    plt.show()

def plot_monthly_heatmap(df, save_dir, last_n_months=24):
    """Generates a heatmap of unemployment rates by region and month."""
    if 'region' not in df.columns or 'period' not in df.columns: return

    pivot_data = df.pivot_table(index='region', columns='period', values='unemployment_rate')
    
    # Limit to the most recent months for clarity
    if len(pivot_data.columns) > last_n_months:
        pivot_data = pivot_data.iloc[:, -last_n_months:]

    plt.figure(figsize=(12, max(5, 0.25 * len(pivot_data.index))))
    sns.heatmap(pivot_data, cmap='mako', linewidths=.5, cbar_kws={'label': 'Unemployment Rate (%)'})
    plt.title(f"Monthly Unemployment Rate Heatmap (Last {last_n_months} Months)", fontweight='bold')
    plt.xlabel("Year-Month")
    plt.ylabel("Region")
    plt.xticks(rotation=45)
    plt.savefig(save_dir / "04_region_month_heatmap.png")
    plt.show()

def plot_lpr_vs_unemployment(df, save_dir):
    """Creates a scatter plot to show the relationship between LPR and unemployment."""
    if 'labour_participation_rate' not in df.columns: return

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x='labour_participation_rate', y='unemployment_rate', hue='area', alpha=0.6, s=50)
    plt.title("Labour Participation Rate vs. Unemployment Rate", fontweight='bold')
    plt.xlabel("Labour Participation Rate (%)")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend(title='Area')
    plt.savefig(save_dir / "05_lpr_vs_unemployment_scatter.png")
    plt.show()


# --- Main Execution ---

def main():
    """Main function to orchestrate the analysis."""
    # Create output directory
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Outputs will be saved to '{RESULTS_DIR}/'")
    
    # 1. Load Data
    raw_df = get_unemployment_data()

    # 2. Clean and Process Data
    df_clean = clean_and_process_data(raw_df)
    
    # 3. Initial Data Overview
    print("\n--- Data Overview ---")
    print("First 5 rows of cleaned data:")
    print(df_clean.head())
    print("\nData types and missing values:")
    df_clean.info()
    
    # 4. Generate Visualizations
    print("\n--- Generating Visualizations ---")
    plot_national_trend(df_clean, RESULTS_DIR)
    plot_top_regions_bar(df_clean, RESULTS_DIR, n=10)
    plot_regional_trends(df_clean, RESULTS_DIR, n=7)
    plot_monthly_heatmap(df_clean, RESULTS_DIR, last_n_months=24)
    plot_lpr_vs_unemployment(df_clean, RESULTS_DIR)
    
    # 5. Save Cleaned Data
    cleaned_filepath = RESULTS_DIR / "cleaned_unemployment_data.csv"
    df_clean.to_csv(cleaned_filepath, index=False)
    print(f"\nCleaned data successfully saved to: {cleaned_filepath}")
    
    # 6. Textual Summaries
    print("\n--- Key Summaries ---")
    if 'region' in df_clean.columns:
        print("Regions with Highest Average Unemployment:")
        region_means = df_clean.groupby('region')['unemployment_rate'].mean().sort_values(ascending=False)
        print(region_means.head(10).to_string())

    if 'period' in df_clean.columns:
        covid_data = df_clean[df_clean['period'].str.contains("2020", na=False)]
        if not covid_data.empty:
            print("\nUnemployment Averages during 2020 (COVID-19 Spike):")
            covid_means = covid_data.groupby('region')['unemployment_rate'].mean().sort_values(ascending=False)
            print(covid_means.head(10).to_string())
    
    print("\nAnalysis complete. All artifacts are in the 'outputs' folder.")

if __name__ == "__main__":
    main()
