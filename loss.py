import numpy as np

def mse(y_true, y_pred):
    error = y_true - y_pred
    
    squared_error = error ** 2
    
    return np.mean(squared_error)