"""
Main entry point for NASA SBIR Investment Analysis.

Research Question:
    Which project themes are most strongly associated with 
    NASA SBIR award amounts?

This script orchestrates the full analysis pipeline:
    1. Data loading and cleaning
    2. Text preprocessing (NLP)
    3. Feature engineering (TF-IDF)
    4. Model training (clustering + regression)
    5. Evaluation and visualization

Usage:
    python main.py

Author: Maia Mailander
Course: Introduction to Data Science and Advanced Programming
Date: December 2025
"""

import os

from src.data_loader import load_and_clean_data, load_processed_data
from src.text_processor import process_abstracts
from src.feature_engineering import create_tfidf_features, prepare_features_for_modeling
from src.models import run_thematic_analysis_pipeline
from src.evaluation import run_evaluation


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
        print("No processed data found. Running initial data cleaning...")
        df = load_and_clean_data()
    
    print(f"Loaded {len(df):,} NASA SBIR awards")
    
    # =========================================================================
    # STEP 2: TEXT PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TEXT PREPROCESSING")
    print("=" * 70)
    
    df = process_abstracts(df)
    
    # Show a sample of cleaned text
    print("\nSample cleaned abstract:")
    print(df['abstract_clean'].iloc[0][:200] + "...")
    
    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    print(f"\nFeature matrix ready: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # =========================================================================
    # STEP 4: MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    
    results = run_thematic_analysis_pipeline(
        df, X, y, tfidf_matrix, vectorizer, feature_names
    )
    
    # =========================================================================
    # STEP 5: EVALUATION AND VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION AND VISUALIZATION")
    print("=" * 70)
    
    results = run_evaluation(results, df, feature_names)
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("""
    Outputs:
    - Console: Full analysis results
    - results/: Plots and summary tables
    
    Next steps:
    - Review plots in results/ folder
    - Use summary tables for your report
    """)
    
    return results


if __name__ == "__main__":
    results = main()