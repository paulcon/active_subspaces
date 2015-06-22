import numpy as np

def load_test_npz(filename):
    return np.load('active_subspaces/tests/data/' + filename)
