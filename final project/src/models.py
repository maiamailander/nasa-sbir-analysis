"""
Machine Learning Models for NASA SBIR Funding Analysis.

METHODOLOGICAL APPROACH:
========================
This analysis focuses on THEMATIC factors that influence NASA funding.
We explicitly separate and exclude structural variables (phase, year, 
company demographics) to isolate the relationship between project 
CONTENT and funding levels.

This module implements a three-part analysis pipeline:

Part 1: UNSUPERVISED LEARNING (K-Means Clustering)
    - Discover natural project groupings from abstract text
    - Identify what types of projects NASA funds

Part 2: DESCRIPTIVE ANALYSIS (Funding by Theme)
    - Calculate average funding per thematic cluster
    - Identify which project types receive highest investment

Part 3: PREDICTIVE MODELING (Text Features Only)
    - Compare 4 regression models using ONLY thematic features
    - Determine which themes predict higher funding
    - Feature importance reveals NASA's thematic priorities

Author: Maia Mailander
Course: Introduction to Data Science and Advanced Programming
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
# FEATURE SEPARATION: STRUCTURAL VS THEMATIC
# =============================================================================

def separate_features(feature_names, verbose=True):
    """
    Programmatically separate structural and thematic features.
    """
    
    # Define structural features - these are EXCLUDED from thematic analysis
    # NOTE: phase_2 no longer exists - phase was dropped in data_loader
    STRUCTURAL_FEATURES = {
        'award year': 'Year the award was granted',
        'woman_owned': 'Whether company is woman-owned',
        'disadvantaged': 'Whether company is socially/economically disadvantaged',
        'number employees': 'Company size',
        'cluster': 'Cluster assignment (if added)'
    }
    
    # ... rest of function stays the same
    
    # Separate features using list comprehension
    structural = [f for f in feature_names if f in STRUCTURAL_FEATURES]
    thematic = [f for f in feature_names if f not in STRUCTURAL_FEATURES]
    
    # Verify thematic features are TF-IDF (text-based)
    tfidf_features = [f for f in thematic if f.startswith('tfidf_')]
    other_thematic = [f for f in thematic if not f.startswith('tfidf_')]
    
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE SEPARATION: Structural vs Thematic")
        print("=" * 70)
        
        print("\n" + "-" * 70)
        print("STRUCTURAL FEATURES (Excluded from thematic analysis)")
        print("-" * 70)
        print("These represent project STRUCTURE, not CONTENT:\n")
        
        for feature in structural:
            description = STRUCTURAL_FEATURES.get(feature, 'No description')
            print(f"  ✗ {feature:<25} | {description}")
        
        print(f"\n  Total structural features: {len(structural)}")
        
        print("\n" + "-" * 70)
        print("THEMATIC FEATURES (Used in thematic analysis)")
        print("-" * 70)
        print("These represent project CONTENT derived from abstracts:\n")
        
        print(f"  ✓ TF-IDF text features: {len(tfidf_features)}")
        print(f"    (Words/phrases extracted from project abstracts)")
        
        if other_thematic:
            print(f"\n  Other thematic features: {len(other_thematic)}")
            for f in other_thematic:
                print(f"    ✓ {f}")
        
        print(f"\n  Total thematic features: {len(thematic)}")
        
        print("\n" + "-" * 70)
        print("RATIONALE")
        print("-" * 70)
        print("""
  Why exclude structural features?
  
  Structural features (especially Phase) are highly predictive of award
  amount, but they don't tell us about NASA's THEMATIC priorities.
  
  Phase II awards are structurally larger than Phase I (~$270K more).
  If we include Phase, it dominates the model and obscures thematic
  patterns.
  
  By analyzing ONLY thematic features, we can answer:
  "Which project TOPICS are associated with higher funding?"
  
  This is distinct from asking:
  "What predicts award amount?" (Answer: Phase)
        """)
    
    return {
        'thematic': thematic,
        'structural': structural,
        'tfidf': tfidf_features,
        'summary': {
            'n_thematic': len(thematic),
            'n_structural': len(structural),
            'n_tfidf': len(tfidf_features)
        }
    }


def create_thematic_feature_matrix(X, feature_names):
    """
    Create a feature matrix containing ONLY thematic features.
    
    This function filters out structural variables, creating a clean
    dataset for thematic analysis.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Full feature matrix
    feature_names : list
        All feature names
        
    Returns:
    --------
    X_thematic : pd.DataFrame
        Feature matrix with only thematic features
    thematic_features : list
        Names of thematic features
    separation_report : dict
        Details of feature separation
    """
    # Separate features
    separation = separate_features(feature_names, verbose=True)
    
    # Filter to thematic features only
    thematic_features = separation['thematic']
    
    # Handle case where X is DataFrame or numpy array
    if isinstance(X, pd.DataFrame):
        X_thematic = X[thematic_features].copy()
    else:
        # Convert to DataFrame first
        X_df = pd.DataFrame(X, columns=feature_names)
        X_thematic = X_df[thematic_features].copy()
    
    print("\n" + "-" * 70)
    print("THEMATIC FEATURE MATRIX CREATED")
    print("-" * 70)
    print(f"  Original matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  Thematic matrix: {X_thematic.shape[0]:,} samples × {X_thematic.shape[1]} features")
    print(f"  Features removed: {X.shape[1] - X_thematic.shape[1]} structural variables")
    
    return X_thematic, thematic_features, separation


# =============================================================================
# PART 1: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# =============================================================================

def find_optimal_clusters(tfidf_matrix, max_k=15, random_state=0):
    """
    Use the Elbow Method to find optimal number of clusters.
    
    The Elbow Method plots inertia (within-cluster sum of squares) against K.
    The "elbow" point where improvement slows down suggests optimal K.
    """
    print("Finding optimal number of clusters (Elbow Method)...")
    print("Inertia = within-cluster sum of squares (lower = tighter clusters)\n")
    
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
    Perform K-Means clustering on project abstracts.
    
    K-Means is an UNSUPERVISED learning algorithm:
    - No labels needed - discovers structure in data
    - Groups similar projects together based on text content
    - Each cluster represents a "type" of NASA project
    """
    print(f"\n" + "-" * 50)
    print(f"TRAINING K-MEANS CLUSTERING MODEL")
    print(f"-" * 50)
    print(f"K-Means is unsupervised: no labels, discovers structure")
    print(f"Parameters: K={n_clusters} clusters")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    print(f"\nTraining... (fitting {tfidf_matrix.shape[0]:,} samples)")
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    print(f"Training complete!")
    
    # Report cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        pct = count / len(cluster_labels) * 100
        bar = "█" * int(pct / 2)
        print(f"  Cluster {cluster}: {count:>5,} projects ({pct:>5.1f}%) {bar}")
    
    return cluster_labels, kmeans


def get_cluster_top_terms(kmeans, vectorizer, n_terms=10):
    """
    Get the top terms that define each cluster.
    
    These terms help us interpret what each cluster represents.
    """
    print(f"\n" + "-" * 50)
    print(f"CLUSTER INTERPRETATION")
    print(f"-" * 50)
    print(f"Top {n_terms} terms per cluster (what defines each group):\n")
    
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    
    for i, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_terms[i] = top_terms
        print(f"  Cluster {i}: {', '.join(top_terms)}")
    
    return cluster_terms


# =============================================================================
# PART 2: DESCRIPTIVE ANALYSIS - FUNDING BY THEME
# =============================================================================

def analyze_funding_by_theme(df, cluster_labels, cluster_terms):
    """
    Analyze average funding by project theme (cluster).
    
    This directly answers: "Which project types get more funding?"
    Pure descriptive analysis — no modeling, just observed patterns.
    """
    print("\n" + "=" * 70)
    print("PART 2: FUNDING BY PROJECT THEME")
    print("=" * 70)
    print("\nQuestion: Which project types receive higher funding from NASA?")
    
    # Add cluster labels
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # Calculate statistics per cluster
    cluster_stats = df_analysis.groupby('cluster')['award amount'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('count', 'count'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(0)
    
    cluster_stats = cluster_stats.sort_values('mean', ascending=False)
    
    # Display results
    print("\n" + "-" * 70)
    print("AVERAGE FUNDING BY PROJECT TYPE")
    print("-" * 70)
    print(f"\n{'Cluster':<8}{'Theme':<40}{'Mean Award':>14}{'Projects':>10}")
    print("-" * 70)
    
    for cluster_id in cluster_stats.index:
        theme = ', '.join(cluster_terms[cluster_id][:3])
        mean_award = cluster_stats.loc[cluster_id, 'mean']
        count = int(cluster_stats.loc[cluster_id, 'count'])
        print(f"{cluster_id:<8}{theme:<40}${mean_award:>13,.0f}{count:>10,}")
    
    # Overall statistics
    overall_mean = df_analysis['award amount'].mean()
    overall_median = df_analysis['award amount'].median()
    
    print("-" * 70)
    print(f"{'OVERALL':<8}{'':<40}${overall_mean:>13,.0f}{len(df_analysis):>10,}")
    
    # Comparison to overall average
    print("\n" + "-" * 70)
    print("COMPARISON TO OVERALL AVERAGE")
    print("-" * 70)
    print(f"\nOverall mean: ${overall_mean:,.0f}")
    print(f"Overall median: ${overall_median:,.0f}\n")
    
    for cluster_id in cluster_stats.index:
        theme = ', '.join(cluster_terms[cluster_id][:2])
        mean_award = cluster_stats.loc[cluster_id, 'mean']
        diff = mean_award - overall_mean
        pct_diff = (diff / overall_mean) * 100
        
        if diff > 0:
            symbol = "▲"
            status = f"+${diff:,.0f} (+{pct_diff:.1f}%)"
        else:
            symbol = "▼"
            status = f"-${abs(diff):,.0f} ({pct_diff:.1f}%)"
        
        print(f"  {symbol} {theme:<35} {status}")
    
    # Key findings
    print("\n" + "-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    
    max_cluster = cluster_stats['mean'].idxmax()
    min_cluster = cluster_stats['mean'].idxmin()
    
    max_theme = ', '.join(cluster_terms[max_cluster][:3])
    min_theme = ', '.join(cluster_terms[min_cluster][:3])
    
    max_mean = cluster_stats.loc[max_cluster, 'mean']
    min_mean = cluster_stats.loc[min_cluster, 'mean']
    
    funding_gap = max_mean - min_mean
    funding_ratio = max_mean / min_mean
    
    print(f"\n  HIGHEST FUNDED: Cluster {max_cluster}")
    print(f"    Theme: {max_theme}")
    print(f"    Average award: ${max_mean:,.0f}")
    
    print(f"\n  LOWEST FUNDED: Cluster {min_cluster}")
    print(f"    Theme: {min_theme}")
    print(f"    Average award: ${min_mean:,.0f}")
    
    print(f"\n  FUNDING GAP: ${funding_gap:,.0f}")
    print(f"    Highest-funded projects receive {funding_ratio:.1f}x more than lowest-funded")
    
    return cluster_stats


# =============================================================================
# PART 3: PREDICTIVE MODELING - THEMATIC FEATURES ONLY
# =============================================================================

def prepare_thematic_regression_data(X_thematic, y, test_size=0.2, random_state=0):
    """
    Prepare data for thematic regression analysis.
    
    Uses ONLY thematic features (TF-IDF text features).
    Structural variables have already been removed.
    """
    print("\n" + "-" * 50)
    print("PREPARING DATA FOR THEMATIC REGRESSION")
    print("-" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_thematic, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split (random_state={random_state}):")
    print(f"  Training: {len(X_train):,} samples ({100-test_size*100:.0f}%)")
    print(f"  Test:     {len(X_test):,} samples ({test_size*100:.0f}%)")
    print(f"  Features: {X_train.shape[1]} (all thematic, no structural)")
    
    # Scale features
    print(f"\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Complete.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_linear_regression(X_train, y_train):
    """Train Linear Regression (baseline, no regularization)."""
    print("\n" + "-" * 50)
    print("TRAINING MODEL 1: LINEAR REGRESSION")
    print("-" * 50)
    print(f"Type: Ordinary Least Squares")
    print(f"Features: {X_train.shape[1]} (thematic only)")
    
    model = LinearRegression()
    print(f"\nTraining...")
    model.fit(X_train, y_train)
    print(f"Complete!")
    
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0):
    """Train Ridge Regression (L2 regularization)."""
    print("\n" + "-" * 50)
    print("TRAINING MODEL 2: RIDGE REGRESSION")
    print("-" * 50)
    print(f"Type: Linear regression with L2 regularization")
    print(f"Alpha: {alpha}")
    print(f"Features: {X_train.shape[1]} (thematic only)")
    
    model = Ridge(alpha=alpha)
    print(f"\nTraining...")
    model.fit(X_train, y_train)
    print(f"Complete!")
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=0):
    """Train Random Forest (ensemble of decision trees)."""
    print("\n" + "-" * 50)
    print("TRAINING MODEL 3: RANDOM FOREST")
    print("-" * 50)
    print(f"Type: Ensemble (bagging)")
    print(f"Trees: {n_estimators}")
    print(f"Features: {X_train.shape[1]} (thematic only)")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    print(f"\nTraining...")
    model.fit(X_train, y_train)
    print(f"Complete! Built {len(model.estimators_)} trees")
    
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=0):
    """Train Gradient Boosting (sequential ensemble)."""
    print("\n" + "-" * 50)
    print("TRAINING MODEL 4: GRADIENT BOOSTING")
    print("-" * 50)
    print(f"Type: Ensemble (boosting)")
    print(f"Stages: {n_estimators}")
    print(f"Features: {X_train.shape[1]} (thematic only)")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=random_state
    )
    
    print(f"\nTraining...")
    model.fit(X_train, y_train)
    print(f"Complete! Built {len(model.estimators_)} stages")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance."""
    print(f"\n" + "-" * 50)
    print(f"EVALUATING: {model_name}")
    print("-" * 50)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    results = {
        'model': model_name,
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }
    
    print(f"\n  {'Metric':<12} {'Train':>12} {'Test':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'R²':<12} {results['train_r2']:>12.4f} {results['test_r2']:>12.4f}")
    print(f"  {'MAE':<12} ${results['train_mae']:>11,.0f} ${results['test_mae']:>11,.0f}")
    print(f"  {'RMSE':<12} ${results['train_rmse']:>11,.0f} ${results['test_rmse']:>11,.0f}")
    
    # Overfitting check
    r2_gap = results['train_r2'] - results['test_r2']
    if r2_gap > 0.1:
        print(f"\n  ⚠️  Possible overfitting (gap = {r2_gap:.3f})")
    else:
        print(f"\n  ✓ Good generalization (gap = {r2_gap:.3f})")
    
    return results


def cross_validate_thematic_models(X_thematic, y, random_state=0):
    """
    Cross-validate all models using ONLY thematic features.
    """
    print("\n" + "-" * 50)
    print("CROSS-VALIDATION (5-fold, Thematic Features Only)")
    print("-" * 50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_thematic)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=10,
            random_state=random_state, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=random_state
        )
    }
    
    cv_results = {}
    
    print(f"\nNote: Using {X_thematic.shape[1]} thematic features (no structural)\n")
    
    for name, model in models.items():
        r2_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        mae_scores = -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        
        cv_results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std()
        }
        
        print(f"{name}:")
        print(f"  R² = {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
        print(f"  MAE = ${mae_scores.mean():,.0f} (+/- ${mae_scores.std():,.0f})\n")
    
    return cv_results


def get_thematic_feature_importance(model, feature_names, top_n=25):
    """
    Extract feature importance for thematic features.
    
    This directly answers: "Which themes predict higher funding?"
    """
    print("\n" + "-" * 70)
    print("THEMATIC FEATURE IMPORTANCE")
    print("-" * 70)
    print("Which themes/words are associated with HIGHER funding?\n")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Clean up feature names for display
    importance_df['theme'] = importance_df['feature'].str.replace('tfidf_', '')
    
    print(f"{'Rank':<6}{'Theme':<25}{'Importance':<12}{'Visual'}")
    print("-" * 70)
    
    for rank, (i, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
        bar = "█" * int(row['importance'] * 200)
        print(f"{rank:<6}{row['theme']:<25}{row['importance']:<12.4f}{bar}")
    
    return importance_df


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def run_thematic_analysis_pipeline(df, X, y, tfidf_matrix, vectorizer, feature_names, random_state=0):
    """
    Run the complete THEMATIC analysis pipeline.
    
    This pipeline focuses exclusively on thematic factors:
    - Structural variables (phase, year, demographics) are explicitly removed
    - Analysis reveals which project TOPICS are associated with funding levels
    
    Pipeline:
        Part 1: Unsupervised Learning - Discover project types
        Part 2: Descriptive Analysis - Funding by theme
        Part 3: Predictive Modeling - Thematic predictors
    """
    print("\n" + "=" * 70)
    print("NASA SBIR THEMATIC ANALYSIS PIPELINE")
    print("=" * 70)
    print("""
    Research Question:
    "Which project THEMES are associated with higher NASA funding?"
    
    Approach:
    We analyze the relationship between project CONTENT (from abstracts)
    and funding levels, explicitly excluding structural factors.
    """)
    
    results = {}
    
    # =========================================================================
    # FEATURE SEPARATION (Key Methodological Step)
    # =========================================================================
    X_thematic, thematic_features, separation = create_thematic_feature_matrix(
        X, feature_names
    )
    results['feature_separation'] = separation
    
    # =========================================================================
    # PART 1: UNSUPERVISED LEARNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: UNSUPERVISED LEARNING")
    print("Research Question: What types of projects does NASA fund?")
    print("=" * 70)
    
    # Elbow method
    k_range, inertias = find_optimal_clusters(tfidf_matrix, max_k=12)
    results['elbow'] = {'k_range': k_range, 'inertias': inertias}
    
    # Clustering
    cluster_labels, kmeans = perform_clustering(tfidf_matrix, n_clusters=8)
    results['clustering'] = {'labels': cluster_labels, 'model': kmeans}
    
    # Interpretation
    cluster_terms = get_cluster_top_terms(kmeans, vectorizer)
    results['cluster_terms'] = cluster_terms
    
    # =========================================================================
    # PART 2: DESCRIPTIVE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: DESCRIPTIVE ANALYSIS")
    print("Research Question: Which project types receive more funding?")
    print("=" * 70)
    
    cluster_funding = analyze_funding_by_theme(df, cluster_labels, cluster_terms)
    results['cluster_funding'] = cluster_funding
    
    # =========================================================================
    # PART 3: PREDICTIVE MODELING (Thematic Features Only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: PREDICTIVE MODELING (Thematic Features Only)")
    print("Research Question: Which themes PREDICT higher funding?")
    print("=" * 70)
    print("\nNote: Structural features (phase, year, etc.) have been EXCLUDED")
    print("      to isolate thematic effects on funding.")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_thematic_regression_data(
        X_thematic, y, random_state=random_state
    )
    
    # Train models
    linear_model = train_linear_regression(X_train, y_train)
    linear_results = evaluate_model(linear_model, X_train, X_test, y_train, y_test, "Linear Regression")
    
    ridge_model = train_ridge_regression(X_train, y_train)
    ridge_results = evaluate_model(ridge_model, X_train, X_test, y_train, y_test, "Ridge Regression")
    
    rf_model = train_random_forest(X_train, y_train, random_state=random_state)
    rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")
    
    gb_model = train_gradient_boosting(X_train, y_train, random_state=random_state)
    gb_results = evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting")
    
    results['models'] = {
        'linear': {'model': linear_model, 'metrics': linear_results},
        'ridge': {'model': ridge_model, 'metrics': ridge_results},
        'random_forest': {'model': rf_model, 'metrics': rf_results},
        'gradient_boosting': {'model': gb_model, 'metrics': gb_results}
    }
    
    # Cross-validation
    print("\n" + "-" * 50)
    print("MODEL COMPARISON")
    print("-" * 50)
    cv_results = cross_validate_thematic_models(X_thematic, y, random_state=random_state)
    results['cross_validation'] = cv_results
    
    # Feature importance
    print("\n" + "=" * 70)
    print("THEMATIC PREDICTORS OF FUNDING")
    print("=" * 70)
    
    # Use best model (Random Forest typically)
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['r2_mean'])
    best_model = results['models'][best_model_name.lower().replace(' ', '_')]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importance_df = get_thematic_feature_importance(best_model, thematic_features)
        results['feature_importance'] = importance_df
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("METHODOLOGY")
    print("-" * 70)
    print(f"""
    Features used: {len(thematic_features)} thematic (text-based)
    Features excluded: {len(separation['structural'])} structural
        - {', '.join(separation['structural'])}
    
    Rationale: Structural features (especially Phase) dominate funding
    predictions but don't reveal thematic priorities. By excluding them,
    we isolate the relationship between project CONTENT and funding.
    """)
    
    print("-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    
    # Clustering findings
    print(f"\n1. PROJECT TYPES (Unsupervised Learning)")
    print(f"   NASA funds 8 distinct project types:")
    for i, terms in cluster_terms.items():
        print(f"     Cluster {i}: {', '.join(terms[:3])}")
    
    # Funding by theme
    print(f"\n2. FUNDING BY THEME (Descriptive)")
    max_cluster = cluster_funding['mean'].idxmax()
    min_cluster = cluster_funding['mean'].idxmin()
    print(f"   Highest: Cluster {max_cluster} ({', '.join(cluster_terms[max_cluster][:2])}) - ${cluster_funding.loc[max_cluster, 'mean']:,.0f}")
    print(f"   Lowest:  Cluster {min_cluster} ({', '.join(cluster_terms[min_cluster][:2])}) - ${cluster_funding.loc[min_cluster, 'mean']:,.0f}")
    
    # Predictive modeling
    print(f"\n3. THEMATIC PREDICTORS (Supervised Learning)")
    print(f"   Best model: {best_model_name}")
    print(f"   CV R²: {cv_results[best_model_name]['r2_mean']:.4f}")
    print(f"   CV MAE: ${cv_results[best_model_name]['mae_mean']:,.0f}")
    
    if 'feature_importance' in results:
        print(f"\n   Top thematic predictors of funding:")
        for i, row in results['feature_importance'].head(5).iterrows():
            print(f"     {row['theme']}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_processed_data
    from text_processor import process_abstracts
    from feature_engineering import create_tfidf_features, create_categorical_features, prepare_features_for_modeling
    
    print("=" * 70)
    print("NASA SBIR THEMATIC ANALYSIS")
    print("=" * 70)
    
    # Load and prepare data
    print("\nLoading data...")
    df = load_processed_data()
    df = process_abstracts(df)
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    df = create_categorical_features(df)
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    # Run thematic analysis pipeline
    results = run_thematic_analysis_pipeline(
        df, X, y, tfidf_matrix, vectorizer, feature_names
    )