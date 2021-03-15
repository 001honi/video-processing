import numpy as np

def MAD(m1,m2):
    return np.absolute(np.subtract(m1,m2)).mean()

def MSE(m1,m2):
    return np.square(np.subtract(m1,m2)).mean()
