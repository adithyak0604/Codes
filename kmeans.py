import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
iris=datasets.load_iris()
x=iris.data
print(x)
kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)
cluster_labels=kmeans.labels_
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1],c=cluster_labels,cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x',s=200,label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title("KMeans Clustering for iris dataset")
plt.legend()
plt.show()