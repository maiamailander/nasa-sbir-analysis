"""
Data loading and preprocessing for NASA SBIR awards analysis.

This module handles:
- Loading raw SBIR award data from Excel
- Filtering to NASA awards only
- Dropping irrelevant columns
- Handling missing values
- Saving cleaned data for analysis
"""

import pandas as pd
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_FILE_PATH = 'data/award_data.xlsx'
PROCESSED_FILE_PATH = 'data/processed/award_data_filtered.csv'
AGENCY_NAME = "National Aeronautics and Space Administration"

# Columns to drop - administrative fields and structural features

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
    'woman owned',
    'socially and economically disadvantaged',
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
    'phase',
    'address1',
    'address2',
    'city',
    'state',
    'zip',
    'proposal award date',
    'contract end date',
    'agency',
    'number employees' 
    'topic code' 
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_raw_data(file_path=RAW_FILE_PATH):
    """Load raw data from Excel file."""
    print(f"Loading data from: {file_path}")
    
    df = pd.read_excel(file_path, engine='openpyxl')
    df.columns = df.columns.str.lower()
    
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def filter_by_agency(df, agency_name=AGENCY_NAME):
    """Filter DataFrame to keep only rows from NASA."""
    rows_before = len(df)
    
    df_filtered = df[df['agency'] == agency_name].copy()
    
    rows_after = len(df_filtered)
    pct_kept = rows_after / rows_before * 100
    
    print(f"  Filtered to '{agency_name}'")
    print(f"  Kept {rows_after:,} of {rows_before:,} rows ({pct_kept:.2f}%)")
    
    return df_filtered


def drop_columns(df, cols_to_drop=COLS_TO_DROP):
    """Drop specified columns from DataFrame."""
    cols_before = len(df.columns)
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    cols_after = len(df.columns)
    
    print(f"  Dropped {cols_before - cols_after} columns")
    print("  NOTE: 'phase' column excluded for thematic analysis")
    print(f"  Remaining columns: {list(df.columns)}")
    
    return df


def handle_missing_values(df):
    """Handle missing values in critical columns."""
    rows_before = len(df)
    
    df = df.dropna(subset=['abstract', 'award amount'])
    
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    
    print(f"  Removed {rows_dropped:,} rows with missing abstract or award amount")
    print(f"  Final row count: {rows_after:,}")
    
    return df


def prepare_features(df):
    """
    Prepare features for analysis by handling remaining missing values
    and dropping columns not needed for modeling.
    """

    # Fill missing topic codes with "Unknown"
    if 'topic code' in df.columns:
        missing_topic = df['topic code'].isnull().sum()
        df['topic code'] = df['topic code'].fillna('Unknown')
        print(f"  Filled {missing_topic:,} missing topic codes with 'Unknown'")
    
    # Drop rows with any remaining missing values in critical columns
    critical_cols = ['company', 'award title', 'abstract', 'award amount', 'award year']
    rows_before = len(df)
    df = df.dropna(subset=critical_cols)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped} rows with missing critical values")
    
    print(f"  Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Remaining columns: {list(df.columns)}")
    
    return df

# Calling each separate function and executing in terminal

def load_and_clean_data(raw_path=RAW_FILE_PATH, save_path=PROCESSED_FILE_PATH):
    """
    Main function: Load, clean, and save the SBIR award data.
    """
    print("=" * 60)
    print("DATA LOADING AND CLEANING")
    print("=" * 60)
    
    print("\n[1/5] Loading raw data...")
    df = load_raw_data(raw_path)
    
    print("\n[2/5] Filtering by agency...")
    df = filter_by_agency(df)
    
    print("\n[3/5] Dropping irrelevant columns...")
    df = drop_columns(df)
    
    print("\n[4/5] Handling missing values...")
    df = handle_missing_values(df)
    
    print("\n[5/5] Preparing features...")
    df = prepare_features(df)
    
    # Save cleaned data
    print("\n" + "-" * 60)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Cleaned data saved to: {save_path}")
    print(f"Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("=" * 60)
    
    return df


def load_processed_data(file_path=PROCESSED_FILE_PATH):
    """Load already-processed data from CSV."""
    print(f"Loading processed data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print("\nSample of cleaned data:")
    print(df.head())