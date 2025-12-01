"""
Data Loader for SBIR Award Data
This module:
- Loads raw SBIR award data from Excel
- Filters to NASA awards only for this project's context
- Drops irrelevant columns
- Saves cleaned data for faster calls later
"""
import os
import pandas as pd

RAW_FILE_PATH = 'data/award_data.xlsx'
PROCESSED_FILE_PATH = 'data/processed/award_data_filtered.csv'
AGENCY_NAME = "National Aeronautics and Space Administration"

# Defining columns to drop - not needed for analysis
COLS_TO_DROP = [
    'branch',
    'program', 
    'agency tracking number',
    'contract',
    'solicitation number',
    'solicitation year',
    'solicitation close date',
    'proposal receipt date',
    'date of notification',
    'uei',
    'duns',
    'hubzone owned',
    'company website',
    'contact name',
    'contact title',
    'contact phone',
    'contact email',
    'pi name',
    'pi title',
    'pi phone',
    'pi email',
    'ri name',
    'ri poc name',
    'ri poc phone',
]


def load_raw_data(file_path=RAW_FILE_PATH):
    """Load raw data from Excel file."""
    print(f"Loading data from: {file_path}")
    
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def filter_by_agency(df, agency_name=AGENCY_NAME):
    """Filter raw data to keep only rows from NASA funded projects."""
    rows_before = len(df)
    
    df_filtered = df[df['agency'] == agency_name].copy()
    
    rows_after = len(df_filtered)
    pct_kept = rows_after / rows_before * 100
    
    print(f"  Filtered to '{agency_name}'")
    print(f"  Kept {rows_after:,} of {rows_before:,} rows ({pct_kept:.1f}%)")
    
    return df_filtered


def drop_columns(df, cols_to_drop=COLS_TO_DROP):
    """Drop irrelevant columns from DataFrame."""
    cols_before = len(df.columns)
    
    # errors='ignore' avoids crashes if a column doesn't exist
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    cols_after = len(df.columns)
    
    print(f"  Dropped {cols_before - cols_after} columns")
    print(f"  Remaining columns: {list(df.columns)}")
    
    return df


def handle_missing_values(df):
    """Handle missing values in critical columns."""
    rows_before = len(df)
    
    # Drop rows where abstract or award amount is missing
    # These are essential for our analysis
    df = df.dropna(subset=['abstract', 'award amount'])
    
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    
    print(f"  Removed {rows_dropped:,} rows with missing abstract or award amount")
    print(f"  Final row count: {rows_after:,}")
    
    return df


def load_and_clean_data(raw_path=RAW_FILE_PATH, save_path=PROCESSED_FILE_PATH):
    """
    Main function: Load, clean, and save the SBIR award data.
    This is the primary entry point for data preprocessing.
    """
    print("=" * 60)
    print("DATA LOADING AND CLEANING")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n[1/4] Loading raw data...")
    df = load_raw_data(raw_path)
    
    # Step 2: Filter to NASA only
    print("\n[2/4] Filtering by agency...")
    df = filter_by_agency(df)
    
    # Step 3: Drop irrelevant columns
    print("\n[3/4] Dropping irrelevant columns...")
    df = drop_columns(df)
    
    # Step 4: Handle missing values
    print("\n[4/4] Handling missing values...")
    df = handle_missing_values(df)
    
    # Save cleaned data
    print("\n" + "-" * 60)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Cleaned data saved to: {save_path}")
    print(f"Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("=" * 60)
    
    return df


def load_processed_data(file_path=PROCESSED_FILE_PATH):
    """
    Load already-processed data from CSV.
    Use this when you've already run the cleaning pipeline.
    """
    print(f"Loading processed data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print("\nSample of cleaned data:")
    print(df.head())