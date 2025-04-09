from nltk.metrics import edit_distance,jaccard_distance
t1="Sending"
t2="Sitting"
edit_dist=edit_distance(t1, t2)
jaccard_dist=jaccard_distance(set(t1),set(t2))
print("Edit Distance:",edit_dist)
print("Jaccard Distance:",jaccard_dist)