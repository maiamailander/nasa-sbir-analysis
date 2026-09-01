"""
Evaluation and Visualization

This module :
- Compares regression approaches (clusters vs TF-IDF)
- Generates plots (cluster distribution, funding by cluster, model comparison)
- Saves results to the results folder
- Creates summaries for reporting

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# CONFIGURATION

RESULTS_DIR = 'results'

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}/")


# MODEL COMPARISON

def compare_approaches(cluster_cv, tfidf_cv):
    """
    Compare cluster-based vs TF-IDF-based regression.
    """
    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON: Clusters vs TF-IDF")
    print("=" * 70)
    
    print("""
    Test: Does clustering lose predictive information?
    
    If cluster regression performs similarly to TF-IDF regression,
    then we have not lost information by clustering.
    """)
    
    # Taking the best results from each approach
    best_cluster = max(cluster_cv.keys(), key=lambda x: cluster_cv[x]['r2_mean'])
    best_tfidf = max(tfidf_cv.keys(), key=lambda x: tfidf_cv[x]['r2_mean'])
    
    cluster_r2 = cluster_cv[best_cluster]['r2_mean']
    cluster_mae = cluster_cv[best_cluster]['mae_mean']
    tfidf_r2 = tfidf_cv[best_tfidf]['r2_mean']
    tfidf_mae = tfidf_cv[best_tfidf]['mae_mean']
    
    print(f"\n{'Approach':<25}{'Features':>10}{'CV R²':>12}{'CV MAE':>15}")
    print("-" * 65)
    print(f"{'Cluster Regression':<25}{8:>10}{cluster_r2:>12.4f}${cluster_mae:>14,.0f}")
    print(f"{'TF-IDF Regression':<25}{500:>10}{tfidf_r2:>12.4f}${tfidf_mae:>14,.0f}")
    
    print(f"\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)
    
    if abs(cluster_r2 - tfidf_r2) < 0.02:
        conclusion = "similar"
        interpretation = """
    Both approaches show similar predictive power.
    This validates that clustering doesn't lose information —
    the 8 cluster features capture the same signal as 500 words.
        """
    elif cluster_r2 > tfidf_r2:
        conclusion = "cluster_better"
        interpretation = """
    Cluster features outperform TF-IDF features.
    Grouping words into coherent themes provides better signal
    than treating words independently.
        """
    else:
        conclusion = "tfidf_better"
        interpretation = """
    TF-IDF features slightly outperform cluster features.
    However, both show low R², confirming that thematic content
    alone cannot predict individual award amounts.
        """
    
    print(interpretation)
    
    return {
        'cluster': {'r2': cluster_r2, 'mae': cluster_mae},
        'tfidf': {'r2': tfidf_r2, 'mae': tfidf_mae},
        'conclusion': conclusion
    }


# VISUALIZATION FUNCTIONS

def plot_elbow_curve(results, save=True):
    """
    Plot the elbow curve for K-Means clustering.
    
    This helps justify the choice of K (number of clusters).
    """
    print("\nGenerating elbow curve plot...")
    
    k_range = results['elbow']['k_range']
    inertias = results['elbow']['inertias']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K Selection', fontsize=14, fontweight='bold')
    
    # Mark the chosen K=8
    if 8 in k_range:
        idx = k_range.index(8)
        ax.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Chosen K=8')
        ax.scatter([8], [inertias[idx]], color='red', s=150, zorder=5)
        ax.legend()
    
    ax.set_xticks(k_range)
    plt.tight_layout()
    
    if save:
        ensure_results_dir()
        filepath = os.path.join(RESULTS_DIR, 'elbow_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    
    plt.close()
    return fig

def plot_funding_by_cluster(results, save=True):
    """
    Bar chart showing mean funding by project theme.
    """
    print("\nGenerating funding by cluster plot...")
    
    cluster_funding = results['cluster_funding']
    cluster_terms = results['cluster_terms']
    
    # Prepare data
    clusters = cluster_funding.index.tolist()
    means = cluster_funding['mean'].values
    
    # Create short labels from cluster terms
    labels = [', '.join(cluster_terms[c][:2]) for c in clusters]
    
    # Sort by mean funding
    sorted_indices = np.argsort(means)[::-1]
    clusters_sorted = [clusters[i] for i in sorted_indices]
    means_sorted = means[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("viridis", len(clusters))
    bars = ax.barh(range(len(clusters)), means_sorted, color=colors)
    
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {clusters_sorted[i]}: {labels_sorted[i]}" 
                         for i in range(len(clusters))], fontsize=10)
    ax.set_xlabel('Mean Award Amount ($)', fontsize=12)
    ax.set_title('Mean NASA SBIR Funding by Project Theme', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, means_sorted)):
        ax.text(val + 5000, bar.get_y() + bar.get_height()/2, 
                f'${val:,.0f}', va='center', fontsize=9)
    
    # Add overall average line
    overall_mean = cluster_funding['mean'].mean()
    ax.axvline(x=overall_mean, color='red', linestyle='--', alpha=0.7, 
               label=f'Overall Average: ${overall_mean:,.0f}')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save:
        ensure_results_dir()
        filepath = os.path.join(RESULTS_DIR, 'funding_by_cluster.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    
    plt.close()
    return fig

def plot_model_comparison(results, save=True):
    """
    Bar chart comparing R² scores across models for both approaches.
    """
    print("\nGenerating model comparison plot...")
    
    cluster_cv = results['cluster_cv']
    tfidf_cv = results['tfidf_cv']
    
    # Prepare data
    models = list(cluster_cv.keys())
    cluster_r2 = [cluster_cv[m]['r2_mean'] for m in models]
    tfidf_r2 = [tfidf_cv[m]['r2_mean'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, cluster_r2, width, label='Cluster Features (8)', color='steelblue')
    bars2 = ax.bar(x + width/2, tfidf_r2, width, label='TF-IDF Features (500)', color='darkorange')
    
    ax.set_ylabel('Cross-Validation R²', fontsize=12)
    ax.set_title('Model Performance Comparison: Clusters vs TF-IDF', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        ensure_results_dir()
        filepath = os.path.join(RESULTS_DIR, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    
    plt.close()
    return fig


def plot_cluster_distribution(results, save=True):
    """
    Pie chart showing distribution of projects across clusters.
    """
    print("\nGenerating cluster distribution plot...")
    
    cluster_funding = results['cluster_funding']
    cluster_terms = results['cluster_terms']
    
    # Prepare data
    clusters = cluster_funding.index.tolist()
    counts = cluster_funding['count'].values
    labels = [f"Cluster {c}: {', '.join(cluster_terms[c][:2])}" for c in clusters]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(clusters))
    wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.1f%%',
                                       colors=colors, pctdistance=0.8)
    
    ax.legend(wedges, labels, title="Project Themes", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax.set_title('Distribution of NASA SBIR Projects by Theme', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        ensure_results_dir()
        filepath = os.path.join(RESULTS_DIR, 'cluster_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
    
    plt.close()
    return fig

# SUMMARY TABLES

def create_cluster_summary_table(results, save=True):
    """
    Create a summary table of clusters with themes and funding.
    """
    print("\nCreating cluster summary table...")
    
    cluster_funding = results['cluster_funding']
    cluster_terms = results['cluster_terms']
    
    # Build summary dataframe
    summary_data = []
    for cluster_id in cluster_funding.index:
        summary_data.append({
            'Cluster': cluster_id,
            'Theme': ', '.join(cluster_terms[cluster_id][:3]),
            'Projects': int(cluster_funding.loc[cluster_id, 'count']),
            'Mean Award': cluster_funding.loc[cluster_id, 'mean'],
            'Median Award': cluster_funding.loc[cluster_id, 'median'],
            'Std Dev': cluster_funding.loc[cluster_id, 'std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean Award', ascending=False)
    
    if save:
        ensure_results_dir()
        filepath = os.path.join(RESULTS_DIR, 'cluster_summary.csv')
        summary_df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")
    
    return summary_df


def create_model_performance_table(results, save=True):
    """
    Create a summary table of model performance metrics.
    """
    print("\nCreating model performance table...")
    
    cluster_cv = results['cluster_cv']
    tfidf_cv = results['tfidf_cv']
    
    # Build performance dataframe
    performance_data = []
    
    for model_name in cluster_cv.keys():
        performance_data.append({
            'Model': model_name,
            'Approach': 'Cluster (8 features)',
            'CV R² Mean': cluster_cv[model_name]['r2_mean'],
            'CV R² Std': cluster_cv[model_name]['r2_std'],
            'CV MAE': cluster_cv[model_name]['mae_mean']
        })
    
    for model_name in tfidf_cv.keys():
        performance_data.append({
            'Model': model_name,
            'Approach': 'TF-IDF (500 features)',
            'CV R² Mean': tfidf_cv[model_name]['r2_mean'],
            'CV R² Std': tfidf_cv[model_name]['r2_std'],
            'CV MAE': tfidf_cv[model_name]['mae_mean']
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    if save:
        ensure_results_dir()
        filepath = os.path.join(RESULTS_DIR, 'model_performance.csv')
        performance_df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")
    
    return performance_df

# FINAL SUMMARY AND REPORTING

def print_final_summary(results, df, thematic_features):
    """
    Print the final summary of the analysis.
    """
    cluster_funding = results['cluster_funding']
    cluster_terms = results['cluster_terms']
    comparison = results['comparison']
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("-" * 70)
    print("METHODOLOGY")
    print("-" * 70)
    print(f"""
    Data: {len(df):,} NASA SBIR projects
    Features: {len(thematic_features)} thematic (TF-IDF)
    Structural features: excluded during feature engineering
    """)
    
    print("-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    
    # Finding 1: Project types
    print("\n1. PROJECT TYPES (Unsupervised Learning)")
    print(f"   Discovered {len(cluster_terms)} distinct thematic categories")
    for i, terms in cluster_terms.items():
        print(f"     Cluster {i}: {', '.join(terms[:3])}")
    
    # Finding 2: Funding patterns
    max_cluster = cluster_funding['mean'].idxmax()
    min_cluster = cluster_funding['mean'].idxmin()
    max_mean = cluster_funding.loc[max_cluster, 'mean']
    min_mean = cluster_funding.loc[min_cluster, 'mean']
    
    print("\n2. FUNDING PATTERNS BY MEAN FUNDING (Descriptive Analysis)")
    print(f"   Highest: Cluster {max_cluster} ({', '.join(cluster_terms[max_cluster][:2])}) - ${max_mean:,.0f}")
    print(f"   Lowest:  Cluster {min_cluster} ({', '.join(cluster_terms[min_cluster][:2])}) - ${min_mean:,.0f}")
    print(f"   Gap: ${max_mean - min_mean:,.0f}")
    
    # Finding 3: Predictive power
    print("\n3. PREDICTIVE POWER (Supervised Learning)")
    print(f"   Cluster CV R²: {comparison['cluster']['r2']:.4f}")
    print(f"   TF-IDF CV R²:  {comparison['tfidf']['r2']:.4f}")
    
    # Conclusion
    print("\n" + "-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    print("""
    Thematic patterns in NASA SBIR funding:
    
    1. AGGREGATE LEVEL: Clear differences exist
       - Certain project themes receive higher average funding
       - This represents meaningful thematic preferences
    
    2. INDIVIDUAL LEVEL: Themes cannot predict specific awards
       - CV R² near 0% for both cluster and TF-IDF approaches
       - Within any theme, awards vary widely
    
    3. IMPLICATION: NASA shows thematic preferences in aggregate,
       but individual award amounts depend on factors beyond theme
       (proposal quality, team experience, program priorities)
    """)

# MAIN EVALUATION PIPELINE

def run_evaluation(results, df):
    """
    Run the complete evaluation pipeline.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_thematic_analysis_pipeline()
    df : pd.DataFrame
        The processed DataFrame
    
    Returns:
    --------
    dict
        Updated results with comparison and file paths
    """
    print("\n" + "=" * 70)
    print("EVALUATION AND VISUALIZATION")
    print("=" * 70)
    
    thematic_features = results['thematic_features']

    # Run comparison
    comparison = compare_approaches(results['cluster_cv'], results['tfidf_cv'])
    results['comparison'] = comparison
    
    # Generate all plots
    print("\n" + "-" * 70)
    print("GENERATING PLOTS")
    print("-" * 70)
    
    plot_elbow_curve(results)
    plot_funding_by_cluster(results)
    plot_model_comparison(results)
    plot_cluster_distribution(results)
    
    # Create summary tables
    print("\n" + "-" * 70)
    print("CREATING SUMMARY TABLES")
    print("-" * 70)
    
    cluster_summary = create_cluster_summary_table(results)
    model_performance = create_model_performance_table(results)
    
    # Store tables in results
    results['tables'] = {
        'cluster_summary': cluster_summary,
        'model_performance': model_performance,
    }
    
    # Print final summary
    print_final_summary(results, df, thematic_features)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {RESULTS_DIR}/")
    print("  - elbow_curve.png")
    print("  - funding_by_cluster.png")
    print("  - model_comparison.png")
    print("  - cluster_distribution.png")
    print("  - cluster_summary.csv")
    print("  - model_performance.csv")
    
    return results

# MAIN EXECUTION

if __name__ == "__main__":
    from data_loader import load_processed_data
    from text_processor import process_abstracts
    from feature_engineering import create_tfidf_features, prepare_features_for_modeling
    from models import run_thematic_analysis_pipeline
    
    print("=" * 70)
    print("NASA SBIR ANALYSIS - EVALUATION PIPELINE")
    print("=" * 70)
    
    # Load and process data
    print("\nLoading data...")
    df = load_processed_data()
    df = process_abstracts(df)
    tfidf_matrix, vectorizer = create_tfidf_features(df)
    X, y, feature_names = prepare_features_for_modeling(df, tfidf_matrix, vectorizer)
    
    # Run model pipeline to get results
    print("\nRunning models...")
    results = run_thematic_analysis_pipeline(
        df, X, y, tfidf_matrix, vectorizer, feature_names
    )
    
    # Run evaluation pipeline
    results = run_evaluation(results, df)
