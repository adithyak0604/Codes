import nltk
# Load your own text data (assuming it's a .txt file)
with open('mydata.txt', 'r') as file:
    text_data = file.read()
# Tokenize the text data (split it into words)
tokens = nltk.word_tokenize(text_data)
# Create an NLTK Text object
text = nltk.Text(tokens)
# Generate a concordance for a specific word, e.g., "network"
text.concordance("network", width=80, lines=10)

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# Step 1: Define the documents
documents = [
"Data is the new oil in the world of technology.",
"Big data helps in better decision making.",
"The amount of data generated every day is massive."
]
# Step 2: Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
# Step 3: Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
# Step 4: Extract feature names (unique words)
feature_names = vectorizer.get_feature_names_out()
# Step 5: Create DataFrame for better visualization
df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
# Step 6: Display the TF-IDF values
print("TF-IDF Matrix:\n")
print(df)

# Import necessary modules
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# Step 2: Define Sample Text
text = "NLTK is a powerful library for NLP."
# Step 3: Tokenize Words
tokens = word_tokenize(text)

# Step 4: Perform POS Tagging
pos_tags = pos_tag(tokens)
# Step 5: Print Results
print("POS Tags:", pos_tags)

