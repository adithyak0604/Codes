import nltk
with open('C:/Users/Adithya K/Documents/VS Code/NLP Lab/data.txt','r') as file:
    text_data=file.read()
tokens=nltk.word_tokenize(text_data)
text=nltk.Text(tokens)
text.concordance("network", width=80, lines=10)