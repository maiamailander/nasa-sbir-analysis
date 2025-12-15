"""
NASA SBIR Thematic Funding Analysis
====================================

Main entry point for the analysis pipeline.

Research Question:
    "Which project THEMES are associated with higher NASA SBIR funding?"

Methodology:
    1. Load and clean NASA SBIR award data
    2. Process abstract text (NLP preprocessing)
    3. Engineer features (TF-IDF vectorization)
    4. Run thematic analysis:
       - Part 1: Unsupervised Learning (clustering, descriptive stats)
       - Part 2: Supervised Learning (regression, validation)

Usage:
    python main.py

Author: Maia Mailander
Course: Introduction to Data Science and Advanced Programming
Date: December 2025
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_and_clean_data, load_processed_data
from src.text_processor import process_abstracts
from src.feature_engineering import (
    create_tfidf_features,
    create_categorical_features,
    prepare_features_for_modeling
)
from src.models import run_thematic_analysis_pipeline


def main():
    """
    Main execution function.
    
    Runs the complete NASA SBIR thematic analysis pipeline:
    1. Data loading and cleaning
    2. Text preprocessing
    3. Feature engineering
    4. Thematic analysis (unsupervised + supervised learning)
    """
    
    print("=" * 70)
    print("NASA SBIR THEMATIC FUNDING ANALYSIS")
    print("=" * 70)
    print("""
    Research Question:
    "Which project THEMES are associated with higher NASA SBIR funding?"
    
    This analysis excludes structural factors (phase, year, demographics)
    to isolate the relationship between project content and funding.
    """)
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)
    
    processed_path = 'data/processed/award_data_filtered.csv'
    
    if os.path.exists(processed_path):
        print(f"\nLoading processed data from: {processed_path}")
        df = load_processed_data(processed_path)
    else:
        print("\nProcessed data not found. Running full data cleaning pipeline...")
        df = load_and_clean_data()
    
    print(f"\nDataset: {len(df):,} NASA SBIR projects")
    print(f"Columns: {list(df.columns)}")
    
    # =========================================================================
    # STEP 2: TEXT PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TEXT PREPROCESSING")
    print("=" * 70)
    
    print("\nCleaning abstract text...")
    print("  - Lowercasing")
    print("  - Removing punctuation and numbers")
    print("  - Removing stop words (including custom structural terms)")
    print("  - Lemmatization")
    
    df = process_abstracts(df)
    
    # Show example
    print("\nExample transformation:")
    print(f"  Original: {df['abstract'].iloc[0][:100]}...")
    print(f"  Cleaned:  {df['abstract_clean'].iloc[0][:100]}...")
    
    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    # TF-IDF vectorization
    print("\nCreating TF-IDF features from cleaned abstracts...")
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    
    # Categorical features
    print("\nCreating categorical features...")
    df = create_categorical_features(df)
    
    # Combine features
    print("\nPreparing final feature matrix...")
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    print(f"\nFeature matrix ready:")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target: award amount (mean=${y.mean():,.0f})")
    
    # =========================================================================
    # STEP 4: THEMATIC ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: THEMATIC ANALYSIS")
    print("=" * 70)
    
    results = run_thematic_analysis_pipeline(
        df=df,
        X=X,
        y=y,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        feature_names=feature_names,
        random_state=0
    )
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
    Output Summary:
    - Discovered {len(results['cluster_terms'])} project types via clustering
    - Analyzed funding patterns across themes
    - Tested predictive power of thematic features
    - Validated with TF-IDF comparison
    
    Key files:
    - Data: data/processed/award_data_filtered.csv
    - Source: src/data_loader.py, text_processor.py, feature_engineering.py, models.py
    """)
    
    return results


if __name__ == "__main__":
    results = main()