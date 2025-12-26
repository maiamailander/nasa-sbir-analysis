"""
Machine Learning Models for NASA SBIR Funding Analysis.

METHODOLOGY:
============
This analysis investigates which project themes are associated with 
higher NASA SBIR funding. We have excluded structural factors 
(phase, year, company demographics) to isolate thematic effects.

STRUCTURE:
==========
PART 1: UNSUPERVISED LEARNING
    1a. K-Means Clustering - Discover natural project groupings
    1b. Descriptive Analysis - Funding patterns by theme

PART 2: SUPERVISED LEARNING
    2a. Cluster Regression - Test if project type predicts funding using three regression models
    2b. TF-IDF Regression (Validation) - Compare to raw word approach

Note: Evaluation, visualization, and final reporting are handled in evaluation.py

"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# FEATURE CONFIRMATION
# =============================================================================

def create_thematic_feature_matrix(X, feature_names):
    """
    Confirm feature matrix contains only thematic features.
    
    Since structural features were excluded during feature engineering,
    X already contains only TF-IDF thematic features.
    """
    print("\n" + "-" * 70)
    print("THEMATIC FEATURE MATRIX")
    print("-" * 70)
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]} (all TF-IDF thematic)")
    print("  Structural features: excluded during feature engineering")
    
    if isinstance(X, pd.DataFrame):
        X_thematic = X.copy()
    else:
        X_thematic = pd.DataFrame(X, columns=feature_names)
    
    thematic_features = list(feature_names)
    
    return X_thematic, thematic_features


# =============================================================================
# PART 1a: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# =============================================================================

def find_optimal_clusters(tfidf_matrix, max_k=15, random_state=0):
    """
    Use Elbow Method to find optimal number of clusters.
    """
    print("\nFinding optimal K using Elbow Method...")
    print("(Inertia = within-cluster sum of squares)\n")
    
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(tfidf_matrix)
        inertias.append(kmeans.inertia_)
        print(f"  K={k:>2}: inertia={kmeans.inertia_:,.0f}")
    
    return list(k_range), inertias


def perform_clustering(tfidf_matrix, n_clusters=8, random_state=0):
    """
    Perform K-Means clustering to discover project types.
    """
    print(f"\n" + "-" * 50)
    print(f"K-MEANS CLUSTERING")
    print(f"-" * 50)
    print(f"K-Means is UNSUPERVISED: discovers structure without labels")
    print(f"Parameters: K={n_clusters} clusters")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    print(f"\nFitting {tfidf_matrix.shape[0]:,} samples...")
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    print(f"Complete!")
    
    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(cluster_labels) * 100
        bar = "█" * int(pct / 2)
        print(f"  Cluster {cluster}: {count:>5,} ({pct:>5.1f}%) {bar}")
    
    return cluster_labels, kmeans


def get_cluster_top_terms(kmeans, vectorizer, n_terms=10):
    """
    Extract top terms defining each cluster.
    """
    print(f"\n" + "-" * 50)
    print(f"CLUSTER THEMES")
    print(f"-" * 50)
    print(f"Top {n_terms} terms per cluster:\n")
    
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    
    for i, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_terms[i] = top_terms
        print(f"  Cluster {i}: {', '.join(top_terms)}")
    
    return cluster_terms


# =============================================================================
# PART 1b: UNSUPERVISED LEARNING - DESCRIPTIVE ANALYSIS
# =============================================================================

def analyze_funding_by_theme(df, cluster_labels, cluster_terms):
    """
    Descriptive analysis of funding patterns by cluster.
    """
    print("\n" + "-" * 50)
    print("FUNDING BY PROJECT THEME")
    print("-" * 50)
    print("\nQuestion: Which project types receive higher funding?\n")
    
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Calculate stats per cluster
    cluster_stats = df_analysis.groupby('cluster')['award amount'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('count', 'count'),
        ('std', 'std'),
    ]).round(0)
    
    cluster_stats = cluster_stats.sort_values('mean', ascending=False)
    
    # Display
    print(f"{'Cluster':<8}{'Theme':<40}{'Mean Award':>14}{'Projects':>10}")
    print("-" * 72)
    
    for cluster_id in cluster_stats.index:
        theme = ', '.join(cluster_terms[cluster_id][:3])
        mean_award = cluster_stats.loc[cluster_id, 'mean']
        count = int(cluster_stats.loc[cluster_id, 'count'])
        print(f"{cluster_id:<8}{theme:<40}${mean_award:>13,.0f}{count:>10,}")
    
    overall_mean = df_analysis['award amount'].mean()
    print("-" * 72)
    print(f"{'OVERALL':<8}{'':<40}${overall_mean:>13,.0f}{len(df_analysis):>10,}")
    
    # Comparison to average
    print(f"\n" + "-" * 50)
    print("COMPARISON TO OVERALL AVERAGE")
    print("-" * 50)
    print(f"\nOverall mean: ${overall_mean:,.0f}\n")
    
    for cluster_id in cluster_stats.index:
        theme = ', '.join(cluster_terms[cluster_id][:2])
        mean_award = cluster_stats.loc[cluster_id, 'mean']
        diff = mean_award - overall_mean
        pct = (diff / overall_mean) * 100
        
        symbol = "▲" if diff > 0 else "▼"
        sign = "+" if diff > 0 else "-"
        print(f"  {symbol} {theme:<35} {sign}${abs(diff):,.0f} ({sign}{abs(pct):.1f}%)")
    
    # Key findings
    max_cluster = cluster_stats['mean'].idxmax()
    min_cluster = cluster_stats['mean'].idxmin()
    max_mean = cluster_stats.loc[max_cluster, 'mean']
    min_mean = cluster_stats.loc[min_cluster, 'mean']
    gap = max_mean - min_mean
    
    print(f"\n" + "-" * 50)
    print("KEY FINDINGS")
    print("-" * 50)
    print(f"\n  Highest: Cluster {max_cluster} ({', '.join(cluster_terms[max_cluster][:2])})")
    print(f"           ${max_mean:,.0f} average")
    print(f"\n  Lowest:  Cluster {min_cluster} ({', '.join(cluster_terms[min_cluster][:2])})")
    print(f"           ${min_mean:,.0f} average")
    print(f"\n  Gap: ${gap:,.0f} ({max_mean/min_mean:.2f}x difference)")
    
    return cluster_stats


# =============================================================================
# PART 2a: SUPERVISED LEARNING - CLUSTER REGRESSION
# =============================================================================

def regression_on_clusters(cluster_labels, y, cluster_terms):
    """
    Test whether project TYPE (cluster) predicts funding.
    
    This is the core supervised learning test: can the thematic
    categories we discovered predict award amounts?
    """
    print("\n" + "-" * 50)
    print("REGRESSION ON PROJECT TYPES")
    print("-" * 50)
    print("""
    Method: Use cluster membership as features (one-hot encoding)
    Question: Does project TYPE predict funding level?
    """)
    
    # One-hot encode clusters
    cluster_dummies = pd.get_dummies(cluster_labels, prefix='cluster')
    print(f"Features: {cluster_dummies.shape[1]} binary cluster indicators")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        cluster_dummies, y, test_size=0.2, random_state=0
    )
    print(f"Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            random_state=0, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=0
        )
    }
    
    # Train and evaluate
    print(f"\n{'Model':<25}{'Train R²':>12}{'Test R²':>12}{'Test MAE':>15}")
    print("-" * 65)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'model': model
        }
        
        print(f"{name:<25}{train_r2:>12.4f}{test_r2:>12.4f}${test_mae:>14,.0f}")
    
    # Cross-validation
    print(f"\n" + "-" * 50)
    print("CROSS-VALIDATION (5-fold)")
    print("-" * 50)
    
    cv_results = {}
    for name, model in [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(alpha=1.0)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0)),
]:
        
        r2_scores = cross_val_score(model, cluster_dummies, y, cv=5, scoring='r2')
        mae_scores = -cross_val_score(model, cluster_dummies, y, cv=5, scoring='neg_mean_absolute_error')
        
        cv_results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean()
        }
        
        print(f"\n{name}:")
        print(f"  CV R² = {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
        print(f"  CV MAE = ${mae_scores.mean():,.0f}")
    
    # Cluster coefficients
    print(f"\n" + "-" * 50)
    print("CLUSTER EFFECTS ON FUNDING")
    print("-" * 50)
    print("\nLinear regression coefficients (effect on award amount):\n")
    
    lr_model = results['Linear Regression']['model']
    coef_df = pd.DataFrame({
        'cluster': cluster_dummies.columns,
        'coefficient': lr_model.coef_
    })
    coef_df['cluster_num'] = coef_df['cluster'].str.replace('cluster_', '').astype(int)
    coef_df['theme'] = coef_df['cluster_num'].apply(
        lambda x: ', '.join(cluster_terms[x][:3])
    )
    coef_df = coef_df.sort_values('coefficient', ascending=False)
    
    print(f"{'Cluster':<8}{'Theme':<40}{'Effect':>15}")
    print("-" * 65)
    
    for _, row in coef_df.iterrows():
        sign = "+" if row['coefficient'] > 0 else ""
        print(f"{row['cluster_num']:<8}{row['theme']:<40}{sign}${row['coefficient']:>14,.0f}")
    
    return results, cv_results, coef_df


# =============================================================================
# PART 2b: VALIDATION - TF-IDF REGRESSION
# =============================================================================

def regression_on_tfidf(X_thematic, y, thematic_features):
    """
    Validation: Compare cluster approach to raw TF-IDF words.
    
    This tests whether our clustering approach loses information
    compared to using all 500 individual word features.
    """
    print("\n" + "-" * 50)
    print("VALIDATION: TF-IDF REGRESSION")
    print("-" * 50)
    print("""
    Purpose: Validate that clustering doesn't lose predictive power
    Method: Compare 8 cluster features vs 500 TF-IDF word features
    """)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_thematic)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=0
    )
    
    print(f"Features: {X_thematic.shape[1]} TF-IDF words")
    print(f"Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=10,
            random_state=0, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=0
        )
    }
    
    # Train and evaluate
    print(f"\n{'Model':<25}{'Train R²':>12}{'Test R²':>12}{'Test MAE':>15}")
    print("-" * 65)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'model': model
        }
        
        flag = " ⚠️ overfitting" if (train_r2 - test_r2) > 0.1 else ""
        print(f"{name:<25}{train_r2:>12.4f}{test_r2:>12.4f}${test_mae:>14,.0f}{flag}")
    
    # Cross-validation
    print(f"\n" + "-" * 50)
    print("CROSS-VALIDATION (5-fold)")
    print("-" * 50)
    
    cv_results = {}
    for name, model in [
        ('Linear Regression', LinearRegression()),
        ('Ridge Regression', Ridge(alpha=1.0)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0, n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0)),
    ]:
        r2_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        mae_scores = -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        
        cv_results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean()
        }
        
        print(f"\n{name}:")
        print(f"  CV R² = {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
        print(f"  CV MAE = ${mae_scores.mean():,.0f}")
    
    # Feature importance
    print(f"\n" + "-" * 50)
    print("TOP PREDICTIVE WORDS")
    print("-" * 50)
    
    rf_model = results['Random Forest']['model']
    importance_df = pd.DataFrame({
        'feature': thematic_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df['word'] = importance_df['feature'].str.replace('tfidf_', '')
    
    print(f"\n{'Rank':<6}{'Word':<25}{'Importance':<12}")
    print("-" * 45)
    
    for rank, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        bar = "█" * int(row['importance'] * 150)
        print(f"{rank:<6}{row['word']:<25}{row['importance']:.4f} {bar}")
    
    return results, cv_results, importance_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_thematic_analysis_pipeline(df, X, y, tfidf_matrix, vectorizer, feature_names, random_state=0):
    """
    Complete thematic analysis pipeline.
    
    PART 1: UNSUPERVISED LEARNING
        1a. K-Means Clustering
        1b. Descriptive Funding Analysis
    
    PART 2: SUPERVISED LEARNING
        2a. Cluster Regression
        2b. TF-IDF Validation
    
    Returns:
        results dict to be passed to evaluation.run_evaluation()
    """
    print("\n" + "=" * 70)
    print("NASA SBIR THEMATIC ANALYSIS")
    print("=" * 70)
    print("""
    Research Question:
    "Which project THEMES are associated with higher NASA funding?"
    
    Methodology:
    - Exclude structural factors (phase, year, demographics)
    - Analyze only thematic content from abstracts
    """)
    
    results = {}
    
    # =========================================================================
    # FEATURE CONFIRMATION
    # =========================================================================
    X_thematic, thematic_features = create_thematic_feature_matrix(
        X, feature_names
    )
    results['thematic_features'] = thematic_features
    
    # =========================================================================
    # PART 1: UNSUPERVISED LEARNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: UNSUPERVISED LEARNING")
    print("=" * 70)
    
    # 1a: Clustering
    print("\n" + "=" * 70)
    print("1a. K-MEANS CLUSTERING")
    print("Research Question: What types of projects does NASA fund?")
    print("=" * 70)
    
    k_range, inertias = find_optimal_clusters(tfidf_matrix, max_k=12)
    results['elbow'] = {'k_range': k_range, 'inertias': inertias}
    
    cluster_labels, kmeans = perform_clustering(tfidf_matrix, n_clusters=8)
    results['clustering'] = {'labels': cluster_labels, 'model': kmeans}
    
    cluster_terms = get_cluster_top_terms(kmeans, vectorizer)
    results['cluster_terms'] = cluster_terms
    
    # 1b: Descriptive Analysis
    print("\n" + "=" * 70)
    print("1b. DESCRIPTIVE ANALYSIS")
    print("Research Question: Which project types receive more funding?")
    print("=" * 70)
    
    cluster_funding = analyze_funding_by_theme(df, cluster_labels, cluster_terms)
    results['cluster_funding'] = cluster_funding
    
    # =========================================================================
    # PART 2: SUPERVISED LEARNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SUPERVISED LEARNING")
    print("=" * 70)
    
    # 2a: Cluster Regression
    print("\n" + "=" * 70)
    print("2a. CLUSTER REGRESSION")
    print("Research Question: Does project TYPE predict funding?")
    print("=" * 70)
    
    cluster_reg, cluster_cv, cluster_coefs = regression_on_clusters(
        cluster_labels, y, cluster_terms
    )
    results['cluster_regression'] = cluster_reg
    results['cluster_cv'] = cluster_cv
    results['cluster_coefficients'] = cluster_coefs
    
    # 2b: TF-IDF Validation
    print("\n" + "=" * 70)
    print("2b. TF-IDF REGRESSION (Validation)")
    print("Purpose: Compare cluster approach to raw word features")
    print("=" * 70)
    
    tfidf_reg, tfidf_cv, tfidf_importance = regression_on_tfidf(
        X_thematic, y, thematic_features
    )
    results['tfidf_regression'] = tfidf_reg
    results['tfidf_cv'] = tfidf_cv
    results['tfidf_importance'] = tfidf_importance
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_processed_data
    from text_processor import process_abstracts
    from feature_engineering import create_tfidf_features, prepare_features_for_modeling
    
    print("=" * 70)
    print("NASA SBIR THEMATIC ANALYSIS - MODEL PIPELINE")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_processed_data()
    df = process_abstracts(df)
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    # Run model pipeline
    results = run_thematic_analysis_pipeline(
        df, X, y, tfidf_matrix, vectorizer, feature_names
    )