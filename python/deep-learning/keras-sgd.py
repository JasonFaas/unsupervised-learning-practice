import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn import datasets
import pandas as pd
from keras.utils import to_categorical

def get_new_model(input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=input_shape))  # input layer 1
    model.add(Dense(100, activation='relu'))  # hidden layer 2
    model.add(Dense(3, activation='softmax'))  # output layer 3
    return model


iris_data = datasets.load_iris()
predictors = iris_data.data
target = to_categorical(iris_data.target)
n_cols = predictors.shape[1]

learning_rates = [.0001, 0.01, 1]

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

for lr in learning_rates:
    print("\n\nModel summary for learning rate of " + str(lr))
    model = get_new_model((n_cols,))
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=2)

    model.fit(predictors, target, validation_split=0.2, epochs=20, callbacks=[early_stopping_monitor])
    model.summary()




