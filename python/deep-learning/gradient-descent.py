import numpy as np
weights = np.array([1, 2])
input_data = np.array([3, 4])
target = 6
learning_rate = 0.01
preds = (weights * input_data).sum()
error = preds - target
print("error: " + str(error))

gradient = 2 * input_data * error
weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print("weights_updated: " + str(weights_updated))
print("error_update: " + str(error_updated))
