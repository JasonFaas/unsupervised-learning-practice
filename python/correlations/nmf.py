from sklearn.decomposition import NMF
from sklearn import datasets

iris_data = datasets.load_iris()
iris_samples = iris_data.data

model = NMF(n_components=1)
model.fit(iris_samples)

nmf_features = model.transform(iris_samples)

print(model.components_)


from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
current_article = norm_features[2,:]
similarities = norm_features.dot(current_article)
print(similarities)
  