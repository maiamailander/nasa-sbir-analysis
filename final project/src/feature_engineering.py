"""
Feature engineering for NASA SBIR analysis.

This module handles:
- TF-IDF vectorization of cleaned abstracts
- Creating numerical features from categorical variables
- Preparing feature matrices for ML models
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(df, text_column='abstract_clean', max_features=500):
    """
    Convert cleaned text to TF-IDF features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with cleaned text column
    text_column : str
        Name of column containing cleaned text
    max_features : int
        Maximum number of words to include (top N by importance)
        
    Returns:
    --------
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix (rows = documents, columns = words)
    vectorizer : TfidfVectorizer
        Fitted vectorizer (needed to inspect feature names)
    """
    print(f"Creating TF-IDF features from '{text_column}'...")
    print(f"  Max features: {max_features}")
    
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,  # Keep top N words
        min_df=5,                   # Word must appear in at least 5 documents
        max_df=0.95,                # Ignore words in >95% of documents
        ngram_range=(1, 2)          # Include single words and two-word phrases
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    # Report results
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  (rows = projects, columns = unique terms)")
    
    # Show top terms by average TF-IDF score
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with numerical features
    """
    print("\nCreating categorical features...")
    
    df = df.copy()
    
    # Phase: Convert to binary (Phase II = 1, Phase I = 0)
    if 'phase' in df.columns:
        df['phase_2'] = (df['phase'].str.contains('II', case=False, na=False)).astype(int)
        print(f"  Created 'phase_2': {df['phase_2'].sum():,} Phase II awards")
    
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with categorical features
    tfidf_matrix : sparse matrix
        TF-IDF features
    vectorizer : TfidfVectorizer
        Fitted vectorizer
        
    Returns:
    --------
    X : pd.DataFrame
        Combined feature matrix
    y : pd.Series
        Target variable (award amount)
    feature_names : list
        Names of all features
    """
    print("\nPreparing final feature matrix...")
    
    # Convert TF-IDF to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()],
        index=df.index
    )
    
    # Select numerical/categorical features
    other_features = df[['phase_2', 'woman_owned', 'disadvantaged', 
                         'number employees', 'award year']].copy()
    
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


# For testing when run directly
if __name__ == "__main__":
    from data_loader import load_processed_data
    from text_processor import process_abstracts
    
    # Load and process data
    df = load_processed_data()
    df = process_abstracts(df)
    
    # Create features
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    df = create_categorical_features(df)
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    print("\nFeature engineering complete!")
    print(f"Ready for modeling with {X.shape[1]} features")