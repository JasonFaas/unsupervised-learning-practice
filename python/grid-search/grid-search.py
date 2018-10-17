from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

param_grid = {'n_neighbors':np.arange(1, 20)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, param_grid, cv=5)


iris_data = datasets.load_iris()
iris_samples = iris_data.data
iris_given_labels = iris_data.target


knn_cv.fit(iris_samples, iris_given_labels)

print(knn_cv.best_params_)
print(knn_cv.best_score_)
