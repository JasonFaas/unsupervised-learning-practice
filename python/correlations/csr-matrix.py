from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD

iris_data = datasets.load_iris()
iris_samples = iris_data.data

model = TruncatedSVD(n_components=1)

model.fit(iris_samples)
# TruncatedSVD(algorithm='randomized')

transformed = model.transform(iris_samples)

print(transformed)