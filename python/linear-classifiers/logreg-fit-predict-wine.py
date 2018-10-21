import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


wine_dataset = sklearn.datasets.load_wine()

X, y = wine_dataset.data, wine_dataset.target
print(X.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=91)


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(logreg.score(x_test, y_test))
print(logreg.predict_proba(X[:1]))


from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(X, y)
print("LinearSVC:" + str(svm.score(X, y)))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X, y)
print("SVC:" + str(svm.score(X, y)))