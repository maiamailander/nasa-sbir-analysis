"""
This module handles:
- Cleaning raw text (lowercase, punctuation removal)
- Removing stop words
- Lemmatization
- Preparing text for TF-IDF vectorization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK data (only needed once)
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
    # Handle missing or non-string values
    if not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Step 3: Split into words
    words = text.split()
    
    # Step 4: Remove stop words
    words = [w for w in words if w not in STOP_WORDS]
    
    # Step 5: Lemmatize (verb tense control)
    words = [LEMMATIZER.lemmatize(w) for w in words]
    
    # Rejoin into single string
    return ' '.join(words)


def process_abstracts(df, text_column='abstract'):
    """
    Process all abstracts in a DataFrame.
    """
    print(f"Processing {len(df):,} abstracts...")
   
    # Create cleaned text column
    df = df.copy()  # Don't modify original
    df['abstract_clean'] = df[text_column].apply(clean_text)
   
    # Report results
    empty_count = (df['abstract_clean'] == '').sum()
    avg_words = df['abstract_clean'].apply(lambda x: len(x.split())).mean()
   
    print(f"  Cleaned {len(df):,} abstracts")
    print(f"  Empty abstracts after cleaning: {empty_count}")
    print(f"  Average words per abstract: {avg_words:.1f}")
   
    return df


# For testing when run directly
if __name__ == "__main__":
    # Test with sample text
    sample = "SpaceWorks proposes an innovative adaptation of its FuseBlox technology for NASA's robotically-assisted operations in cryogenic environments. The system uses 3D-printed components."
    
    print("Original text:")
    print(sample)
    print("\nCleaned text:")
    print(clean_text(sample))