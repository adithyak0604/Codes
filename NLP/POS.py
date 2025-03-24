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
