from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load iris data
iris_data = datasets.load_iris()
iris_samples = iris_data.data
iris_given_labels = iris_data.target


#scaling data before KMeans calculation
scaler = StandardScaler()
scaler.fit(iris_samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
iris_samples = scaler.transform(iris_samples)
# mean is 0 and variance is 1




# setup kmean
model = KMeans(n_clusters=3)
model.fit(iris_samples)

pred_labels = model.predict(iris_samples)

centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,2]

# display samples, clusters and centroids
xs = iris_samples[:,0]
ys = iris_samples[:,2]
plt.scatter(xs,ys, c = iris_given_labels)
plt.scatter(centroids_x, centroids_y, marker='D', s=100)
plt.show()


# display cross tabulation table comparing pred_labels and iris_labels
df = pd.DataFrame({'labels':pred_labels, 'species':iris_given_labels})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)


# measuring clustering quality without initial labels
print("inertia 3:" + str(model.inertia_))
# best cluster count is likely where inertia decreases more slowly
prev_inertia = 10 * 1000
for i in range(1,10):
    model_loop = KMeans(n_clusters=i)
    model_loop.fit(iris_samples)
    print("inertia " + str(i) + ":" + str(int(model_loop.inertia_)))
    print("\tDecrease %:" + str(int(100 - (100 * model_loop.inertia_ / prev_inertia))))
    prev_inertia = model_loop.inertia_
