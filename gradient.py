import numpy as np
from model import predict


def compute_gradients(X, Y, m, b):

    predictions = predict(X, m, b)

    error = Y - predictions

    gradient_m = -(2 / len(X)) * np.sum(X * error)

    gradient_b = -(2 / len(X)) * np.sum(error)

    return gradient_m, gradient_b