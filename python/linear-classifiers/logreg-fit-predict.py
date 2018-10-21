import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()

X, y = newsgroups.data, newsgroups.target
print(X.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=91)


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(logreg.score(x_test, y_test))
