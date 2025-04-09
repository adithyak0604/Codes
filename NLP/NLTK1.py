from nltk import word_tokenize, sent_tokenize
text="Hello Guys, Welcome to NLP Lab"
word_t=word_tokenize(text)
sent_t=sent_tokenize(text)
print("Word Tokenization:",word_t)
print("Sentence Tokenization:",sent_t)