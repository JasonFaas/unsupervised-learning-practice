from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

iris_data = datasets.load_iris()
iris_samples = iris_data.data

model = PCA()
iris_features = model.fit_transform(iris_samples)

for i in range(0,3):
    for k in range(i+1, 4):
        x = iris_features[:,i]
        y = iris_features[:,k]
        print(str(i) + ":" + str(k) + ":" + str(pearsonr(x,y)[0]))

features = range(model.n_components_)
plt.bar(features, model.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

plt.scatter(x, y)
plt.show()


