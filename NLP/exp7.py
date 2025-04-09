from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
document=["Science is the pursuit of knowledge through observation, experimentation, and reasoning."
          "It helps us understand the natural world, from the tiniest particles to the vastness of space."
          "Through science, we've discovered the laws of physics, explored the depths of the oceans, and uncovered the secrets of the human body."]
vectorizer=TfidfVectorizer()
tfidf_matrix=vectorizer.fit_transform(document)
feature_names=vectorizer.get_feature_names_out()
df=pd.DataFrame(tfidf_matrix.toarray(),columns=feature_names)
print("TF-IDF Matrix: \n")
print(df)