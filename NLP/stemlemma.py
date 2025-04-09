from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
stemmer=PorterStemmer()
lemma=WordNetLemmatizer()
word="swimming"
stemmed_word=stemmer.stem(word)
lemmatized_word=lemma.lemmatize(word, pos=wordnet.VERB)
print("Stemmed Word:",stemmed_word)
print("Lemmatized Word:",lemmatized_word)
