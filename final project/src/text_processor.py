"""
Text preprocessing for NASA SBIR abstract analysis.

This module handles:
- Cleaning raw text (lowercase, punctuation removal)
- Removing stop words
- Lemmatization
- Preparing text for TF-IDF vectorization

IMPORTANT: We remove structural/administrative language (e.g., "Phase II",
"NASA", "SBIR") to focus on project content, not structure.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_data():
    """Download required NLTK datasets."""
    resources = ['stopwords', 'wordnet', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)


# Initialize NLTK components
download_nltk_data()
STOP_WORDS = set(stopwords.words('english'))

# =============================================================================
# CUSTOM STOP WORDS
# =============================================================================
# These terms are removed because they reflect project structure or administrative language.
# =============================================================================

CUSTOM_STOP_WORDS = {
    # Phase references - structural, not thematic
    'phase', 'ii', 'iii', 'iv', 
    'one', 'two', 'three', 'four', 'first', 'second', 'third', 'fourth',

    # Program names - administrative

    'sbir', 'sttr',

    # Agency references - not thematic
    'nasa', 'jpl', 'gsfc', 'agency', 'center', 'centers',
    
    # Proposal/project language - generic administrative
    'proposal', 'proposes',
    'effort', 'work', 'program', 'task', 'tasks',
    'objective', 'objectives', 'aim', 'aims',
    
    # Development language - too generic
    'develop', 'developed', 'development', 'developing',
    
    # Future/conditional tense - not content
    'would', 'could', 'should', 'may', 'might', 'can',
    
    # Generic technical terms that appear everywhere
    'technology', 'system', 'systems', 'trl',
    'method', 'approach', 'technique', 'behavior', 
    'research', 'study', 'analysis', 'case',
    'data', 'information', 'resulted',
    'result', 'results', 'finding', 'findings',
    'goal', 'outcome', 'outcomes',
    'innovation', 'innovative', 'novel', 'new',
    
    # Company/team references
    'team', 'company', 'firm', 'inc', 'llc', 'corp', 'user',
    
    # Common filler words
    'also', 'well', 'however', 'therefore', 'thus',
    'using', 'used', 'use', 'based',
    'provides', 'provided', 'providing',
    'enable', 'enables', 'enabled', 'enabling',
    'support', 'supports', 'supported', 'supporting',
    'include', 'includes', 'included', 'including',
    'require', 'requires', 'required', 'requiring', 'our', 'the', 
    'project', 'will', 'be', 
    'is', 'with', 'that', 'this',
    'which', 'we', 'not', 'its', 'from', 'have', 'has',
    'more', 'other', 'such', 'these', 'those', 'than',

    # Additional non-related verbs
    'propose', 'proposed', 'seek', 'seeks',
    'design', 'designed',
    'create', 'created', 'creating',
    'improve', 'improved', 'improving', 'improvement',
    'enhance', 'enhanced', 'enhancing', 'enhancement',
    'provide', 'advanced', 'build', 'building', 'built', 'builds',
    'demonstrate', 'demonstrated', 'demonstrating',
    'utilize', 'utilized', 'utilizing',
    'manage', 'managed', 'management',
    'source', 'sourced', 'sourcing',
    'multiple', 'various', 'different',
    'integrated', 'integration', 'integrate',
    
    # Performance language
    'performance', 'capability', 'capabilities',
    'solution', 'solutions',
    'application', 'applications',
    'extreme', 
    'safe', 'safety',
    'relevant', 'relate', 'related',
    'efficient', 'efficiency', 'optimize', 'optimized',
    'mission', 'missions', 'prototype',
    'successful', 'successfully', 'success',
    'deliver', 'delivered', 'delivering', 'delivery',
    
    # Process language
    'process', 'processes', 'processing',
    'test', 'testing', 'tested', 'tests',
    'evaluate', 'evaluation', 'evaluated',
    'control', 'controlled', 'controlling',
    
    # Cost/benefit language (not thematic)
    'cost', 'costs', 'reduce', 'reduced', 'reduction',
    'benefit', 'benefits', 'high', 'low',
    'commercial', 'commercialization',
    
    # Time references
    'current', 'currently', 'existing',
    'future', 'potential', 'potentially',
    'previously', 'prior', 'historical',
}

# Combine with standard stop words
STOP_WORDS = STOP_WORDS.union(CUSTOM_STOP_WORDS)

LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """
    Clean a single text string.
    
    Parameters:
    str
        Raw abstract text
        
    Returns:
    str
        Cleaned text ready for vectorization
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stop words and short words
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Lemmatize
    words = [LEMMATIZER.lemmatize(w) for w in words]
    
    return ' '.join(words)


def process_abstracts(df, text_column='abstract'):
    """
    Process all abstracts in a DataFrame.
    """
    print(f"Processing {len(df):,} abstracts...")
    print(f"  Removing {len(CUSTOM_STOP_WORDS)} custom stop words (structural/administrative terms)")
    
    df = df.copy()
    df['abstract_clean'] = df[text_column].apply(clean_text)
    
    empty_count = (df['abstract_clean'] == '').sum()
    avg_words = df['abstract_clean'].apply(lambda x: len(x.split())).mean()
    
    print(f"  Cleaned {len(df):,} abstracts")
    print(f"  Empty abstracts after cleaning: {empty_count}")
    print(f"  Average words per abstract: {avg_words:.1f}")
    
    return df

def show_cleaning_example():
    """
    Demonstrate text cleaning with a sample abstract.
    """
    sample = """
    In this Phase II SBIR effort, NASA proposes to develop an innovative
    lunar propulsion system. The team will demonstrate the technology
    using advanced thermal management. This project builds on Phase I results.
    """
    
    print("\n  Original text:")
    print(f"    {sample.strip()}")
    print("\n  Cleaned text:")
    print(f"    {clean_text(sample)}")

if __name__ == "__main__":
    sample = """
    In this Phase II SBIR effort, NASA proposes to develop an innovative
    lunar propulsion system. The team will demonstrate the technology
    using advanced thermal management. This project builds on Phase I results.
    """
    
    print("Original text:")
    print(sample)
    print("\nCleaned text:")
    print(clean_text(sample))