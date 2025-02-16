import numpy as np
import pandas as pd
import pickle
import re
from collections import Counter
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk  # Import NLTK for downloading resources

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to clean text (you need to define this function if it's not already defined)
def clean_text(text):
    """Cleans the input text by removing special characters, numbers, and extra spaces, and converts to lowercase."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

# Function to create vocabulary with POS tags
def create_vocab_with_pos(df):
    """Function to create vocabulary with POS tags from the review DataFrame."""
    # Collect all words and their POS tags
    all_words = []

    for review in df['Text']:
        tokens = word_tokenize(clean_text(review))  # Clean and tokenize the review
        pos_tags = pos_tag(tokens)  # Get POS tags
        all_words.extend([f"{word}_{pos}" for word, pos in pos_tags])  # Append word_POS to the list

    # Create vocabulary from the words
    word_counts = Counter(all_words)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}  # Start indexing from 1
    vocab["<UNK>"] = len(vocab) + 1  # Add <UNK> token

    return vocab

# Load your data
df = pd.read_csv("/content/data/Reviewsfull.csv")  # Ensure your path is correct

vocab2int = create_vocab_with_pos(df)

with open("/content/data/vocab2int.pickle", "wb") as f:
    pickle.dump(vocab2int, f)

print("Vocabulary with POS tags saved.")
