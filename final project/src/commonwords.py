import pandas as pd
from collections import Counter

# Load your processed data
df = pd.read_csv('data/processed/award_data_filtered.csv')

# Check if you have cleaned abstracts already
if 'abstract_clean' in df.columns:
    text_column = 'abstract_clean'
    print("Using cleaned abstracts")
else:
    text_column = 'abstract'
    print("Using raw abstracts (not yet cleaned)")

# Combine all abstracts into one big string
all_text = ' '.join(df[text_column].dropna().astype(str))

# Split into words and count
words = all_text.lower().split()
word_counts = Counter(words)

# Show top 100 most frequent words
print("\nTop 100 Most Frequent Words:\n")
for i, (word, count) in enumerate(word_counts.most_common(100), 1):
    print(f"{i:3}. {word:20} {count:,}")