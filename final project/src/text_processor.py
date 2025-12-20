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
    'phase', 'i', 'ii', 'iii', 'iv',
    
    # Program names - administrative
    'sbir', 'sttr',
    
    # Agency references - not thematic
    'nasa', 'jpl', 'gsfc', 'agency',
    
    # Proposal/project language - generic administrative
    'proposal', 'proposed', 'proposes', 'propose',
    'project', 'effort', 'work', 'program',
    
    # Development language - too generic
    'develop', 'developed', 'development', 'developing',
    'demonstrate', 'demonstrated', 'demonstration', 'demonstrating',
    
    # Future/conditional tense - not content
    'will', 'would', 'could', 'should', 'may', 'might', 'can'
    
    # Generic technical terms that appear everywhere
    'technology', 'system', 'systems',
    'method', 'approach', 'technique',
    'research', 'study', 'analysis',
    'result', 'results',
    'goal', 'objective',
    'innovation', 'innovative', 'novel', 'new',
    
    # Company/team references
    'team', 'company', 'firm', 'inc', 'llc', 'corp',
    
    # Common filler words
    'also', 'well', 'however', 'therefore', 'thus',
    'using', 'used', 'use', 'based',
    'provide', 'provides', 'provided', 'providing',
    'enable', 'enables', 'enabled', 'enabling',
    'support', 'supports', 'supported', 'supporting',
    'include', 'includes', 'included', 'including',
    'require', 'requires', 'required', 'requiring', 'our', 'the', 
    'project', 'and', 'of', 'to', 'a', 'in', 'for', 'will', 'be', 
    'is', 'with', 'that', 'this', 'as', 'on', 'an', 'by', 'are', 'at',
    'which', 'i', 'we', 'not', 'its',

    # Additional non-related verbs
    'propose', 'proposed', 'aim', 'aims', 'seek', 'seeks',
    'design', 'designed',  # Often generic ("designed to...")
    'create', 'created', 'creating',
    'improve', 'improved', 'improving', 'improvement',
    'enhance', 'enhanced', 'enhancing', 'enhancement',
    'develop', 'developed', 'using', 'use',
    'provide'
    
    # Performance language
    'performance', 'capability', 'capabilities',
    'solution', 'solutions',
    'application', 'applications',
    
    # Process language
    'process', 'processes', 'processing',
    'test', 'testing', 'tested',
    'evaluate', 'evaluation', 'evaluated',
    
    # Cost/benefit language (not thematic)
    'cost', 'costs', 'reduce', 'reduced', 'reduction',
    'benefit', 'benefits',
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
    -----------
    text : str
        Raw abstract text
        
    Returns:
    --------
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