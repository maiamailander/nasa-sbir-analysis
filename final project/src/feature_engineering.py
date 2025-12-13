"""
Feature engineering for NASA SBIR analysis.

This module handles:
- TF-IDF vectorization of cleaned abstracts
- Creating numerical features from categorical variables
- Preparing feature matrices for ML models

NOTE: Phase-related features are NOT created here.
      Our analysis focuses on thematic content only.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(df, text_column='abstract_clean', max_features=500):
    """
    Convert cleaned text to TF-IDF features.
    """
    print(f"\nCreating TF-IDF features from '{text_column}'...")
    print(f"  Max features: {max_features}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  (rows = projects, columns = unique terms)")
    
    # Show top terms
    feature_names = vectorizer.get_feature_names_out()
    avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = avg_scores.argsort()[-10:][::-1]
    
    print(f"\n  Top 10 terms by average TF-IDF score:")
    for i in top_indices:
        print(f"    - {feature_names[i]}: {avg_scores[i]:.4f}")
    
    return tfidf_matrix, vectorizer


def create_categorical_features(df):
    """
    Convert categorical columns to numerical features.
    
    NOTE: Phase-related features are NOT created.
          Our analysis focuses on thematic content only.
    """
    print("\nCreating categorical features...")
    print("  NOTE: Phase excluded - focusing on thematic analysis")
    
    df = df.copy()
    
    # Woman owned: Convert Y/N to binary
    if 'woman owned' in df.columns:
        df['woman_owned'] = (df['woman owned'].str.upper() == 'Y').astype(int)
        print(f"  Created 'woman_owned': {df['woman_owned'].sum():,} woman-owned companies")
    
    # Socially/economically disadvantaged: Convert Y/N to binary
    if 'socially and economically disadvantaged' in df.columns:
        df['disadvantaged'] = (df['socially and economically disadvantaged'].str.upper() == 'Y').astype(int)
        print(f"  Created 'disadvantaged': {df['disadvantaged'].sum():,} disadvantaged-owned companies")
    
    return df


def prepare_features_for_modeling(df, tfidf_matrix, vectorizer):
    """
    Combine TF-IDF and categorical features into final feature matrix.
    
    NOTE: Only thematic features are included.
          Structural features (phase, year) are excluded.
    """
    print("\nPreparing final feature matrix...")
    
    # Convert TF-IDF to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()],
        index=df.index
    )
    
    # Select non-structural features only
    # We include demographics and company size as they're not "structural" in the same way as phase
    other_features = df[['woman_owned', 'disadvantaged', 'number employees', 'award year']].copy()
    
    # Combine all features
    X = pd.concat([other_features, tfidf_df], axis=1)
    
    # Target variable
    y = df['award amount']
    
    print(f"  Final feature matrix: {X.shape[0]:,} samples, {X.shape[1]:,} features")
    print(f"  Target variable: award amount")
    print(f"    Mean: ${y.mean():,.0f}")
    print(f"    Median: ${y.median():,.0f}")
    print(f"    Range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    return X, y, list(X.columns)


if __name__ == "__main__":
    from data_loader import load_processed_data
    from text_processor import process_abstracts
    
    df = load_processed_data()
    df = process_abstracts(df)
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    df = create_categorical_features(df)
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    print("\nFeature engineering complete!")
    print(f"Ready for modeling with {X.shape[1]} features")