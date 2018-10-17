from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

iris_data = datasets.load_iris()
iris_samples = iris_data.data
iris_given_labels = iris_data.target

model = KMeans(n_clusters=3)
model.fit(iris_samples)

pred_labels = model.predict(iris_samples)

print(pred_labels)
print(iris_given_labels)

print(len(pred_labels))
labels = pred_labels == iris_given_labels
print(np.count_nonzero(labels))

centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,2]

xs = iris_samples[:,0]
ys = iris_samples[:,2]
plt.scatter(xs,ys, c = iris_given_labels)
plt.scatter(centroids_x, centroids_y, marker='D', s=100)
plt.show()