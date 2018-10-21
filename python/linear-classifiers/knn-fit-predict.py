import sklearn.datasets
from sklearn.model_selection import train_test_split

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()

X, y = newsgroups.data, newsgroups.target
print(X.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=91)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(knn.score(x_test, y_test))
