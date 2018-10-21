import numpy as np

# dot product
x = np.arange(3)
y = np.arange(3, 6)
print(np.sum(x * y))
print(x @ y)


from sklearn import datasets
iris_data = datasets.load_iris()
x = iris_data.data
y = iris_data.target

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x, y)
print(model.predict(x)[0])
print(model.predict(x)[-1])

print(model.coef_ @ x[0] + model.intercept_)
print(model.coef_ @ x[-1] + model.intercept_)

print(model.coef_)
print(model.intercept_)


# minimizing a loss
from scipy.optimize import minimize
print(minimize(np.square, 2).x)