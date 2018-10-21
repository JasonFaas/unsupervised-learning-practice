import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn import datasets
import pandas as pd
from keras.utils import to_categorical

iris_data = datasets.load_iris()
predictors = iris_data.data
target = to_categorical(iris_data.target)

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,))) # input layer 1
model.add(Dense(100, activation='relu')) # hidden layer 2
model.add(Dense(100, activation='relu')) # hidden layer 3
model.add(Dense(3, activation='softmax')) # output layer 4

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target)
model.summary()

exit(0)
from keras.models import load_model
model.save('keras_model_iris_data.h5')
my_model = load_model('keras_model_iris_data.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:, 1]