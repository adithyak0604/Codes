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
