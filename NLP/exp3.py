import nltk
from nltk.probability import FreqDist
text="""Science is the pursuit of knowledge through observation, experimentation, and reasoning.
      It helps us understand the natural world, from the tiniest particles to the vastness of space.
      Through science, we've discovered the laws of physics, explored the depths of the oceans, and uncovered the secrets of the human body."""
tokens=nltk.word_tokenize(text.lower())
total_tokens=len(tokens)
unique_tokens=len(set(tokens))
freq_dist=FreqDist(tokens)
the_count=freq_dist["the"]
the_percentage=(the_count/total_tokens)*100
print(f"1. Total no.of words: {total_tokens}")
print(f"2. Total no.of unique words: {unique_tokens}")
print(f"3. No.of times the word 'the' appers: {the_count}")
print(f"4. Percentage of 'the' in the text: {the_percentage:.2f}%")