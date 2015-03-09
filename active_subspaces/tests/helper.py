import numpy as np

def load_test_npz(filename):
    return np.load('data/' + filename)
