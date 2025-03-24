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
