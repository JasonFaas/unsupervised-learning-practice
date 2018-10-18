from sklearn import datasets
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

iris_data = datasets.load_iris()
iris_samples = iris_data.data

for i in range(0,3):
    for k in range(i+1, 4):
        x = iris_samples[:,i]
        y = iris_samples[:,k]
        print(str(i) + ":" + str(k) + ":" + str(round(pearsonr(x,y)[0], 2)))

plt.scatter(x, y)
plt.show()
