"""
Research Question:
    Do Space Project Themes Influence Funding? A Machine Learning Approach Using K-Means Clustering and Regression Model Comparison

This script runs the full analysis pipeline:
    1. Data loading and cleaning
    2. Text preprocessing 
    3. Feature engineering (TF-IDF)
    4. Model training (clustering + regression)
    5. Evaluation and visualization

Author: Maia Mailander
Course: Introduction to Data Science and Advanced Programming
Date: January 2026
"""

import os

from src.data_loader import load_and_clean_data, load_processed_data
from src.text_processor import process_abstracts, show_cleaning_example
from src.feature_engineering import create_tfidf_features, prepare_features_for_modeling
from src.models import run_thematic_analysis_pipeline
from src.evaluation import run_evaluation


def main():
    """Run the complete analysis pipeline."""
    
    print("=" * 70)
    print("NASA SBIR INVESTMENT ANALYSIS")
    print("Research Question: Do Space Project Themes Influence Funding? A Machine Learning Approach Using K-Means Clustering and Regression Model Comparison.")
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
    
    show_cleaning_example()
    df = process_abstracts(df)
    
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
    print("""
    Part 1: Unsupervised Learning (K-Means Clustering)
        - Discover natural project groupings from TF-IDF features
        - Analyze funding patterns by theme
    
    Part 2: Supervised Learning (Regression)
        - Test if cluster membership predicts funding
        - Compare: Linear, Ridge, Random Forest, Gradient Boosting
    """)
    
    results = run_thematic_analysis_pipeline(
        df, X, y, tfidf_matrix, vectorizer, feature_names
    )
    
    # =========================================================================
    # STEP 5: EVALUATION AND VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION AND VISUALIZATION")
    print("=" * 70)
    print("""
    - Compare cluster vs TF-IDF regression approaches
    - Generate plots (elbow curve, funding by cluster, model comparison)
    - Create summary tables for reporting
    """)
    
    results = run_evaluation(results, df)
    
    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("""
    Key Outputs:
    ├── results/
    │   ├── elbow_curve.png           # K selection justification
    │   ├── funding_by_cluster.png    # Theme vs funding visualization
    │   ├── cluster_wordclouds.png    # Theme visualization
    │   ├── feature_importance.png    # Top predictive words
    │   ├── model_comparison.png      # Regression model comparison
    │   ├── cluster_distribution.png  # Project distribution by theme
    │   ├── cluster_summary.csv       # Theme statistics
    │   ├── model_performance.csv     # CV results for all models
    │   └── top_predictive_words.csv  # Feature importance ranking
    
    Research Finding:
        Thematic patterns exist at aggregate level (laser/lidar projects
        receive ~$48K more than materials projects), but themes cannot
        predict individual award amounts (CV R² ≈ -0.045 for all models).
    """)
    
    return results


if __name__ == "__main__":
    results = main()