"""
Research Question:
    Which project characteristics are most strongly associated with 
    NASA SBIR award amounts?

This script orchestrates the full analysis pipeline:
    1. Data loading and cleaning
    2. Text preprocessing (NLP)
    3. Feature engineering
    4. Model training and comparison
    5. Evaluation and visualization

"""

from src.data_loader import load_and_clean_data, load_processed_data
import os


def main():
    """Run the complete analysis pipeline."""
    
    print("=" * 70)
    print("NASA SBIR INVESTMENT ANALYSIS")
    print("Identifying drivers of funding in the space technology sector")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)
    
    # Check if processed data already exists
    processed_path = 'data/processed/award_data_filtered.csv'
    
    if os.path.exists(processed_path):
        print(f"Found existing processed data at {processed_path}")
        df = load_processed_data(processed_path)
    else:
        print("No processed data found. Running full cleaning pipeline...")
        df = load_and_clean_data()
    
    print(f"\nDataset ready: {len(df):,} projects, {len(df.columns)} features")
    
    # =========================================================================
    # STEP 2: TEXT PREPROCESSING (TODO)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TEXT PREPROCESSING")
    print("=" * 70)
    print("Coming soon...")
    
    # =========================================================================
    # STEP 3: FEATURE ENGINEERING (TODO)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    print("Coming soon...")
    
    # =========================================================================
    # STEP 4: MODEL TRAINING (TODO)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    print("Coming soon...")
    
    # =========================================================================
    # STEP 5: EVALUATION (TODO)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION & RESULTS")
    print("=" * 70)
    print("Coming soon...")
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    df = main()