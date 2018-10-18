# t-distributed stochastic neighbor embedding

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

iris_data = datasets.load_iris()
iris_samples = iris_data.data
species = iris_data.target

for i in range(10, 500, 50):
    print(i)
    model = TSNE(learning_rate=i)
    transformed = model.fit_transform(iris_samples)
    xs = transformed[:,0]
    ys = transformed[:,1]
    plt.scatter(xs, ys, c=species)
    plt.show()
