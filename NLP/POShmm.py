import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
nltk.download('treebank')
nltk.download('universal_tagset')
train_data=treebank.tagged_sents(tagset='universal')[:3000]
test_data=treebank.tagged_sents(tagset='universal')[3000:3100]
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)
sample="The quick brown fox jumps over the lazy dog".split()
tagged_sentence=hmm_tagger.tag(sample)
print("POS Tagged Sentence:")
for word, tag in tagged_sentence:
    print(f"{word} -> {tag}")
