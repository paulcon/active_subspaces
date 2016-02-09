import numpy as np
import active_subspaces
import matplotlib.pyplot as plt

# Load the data 
npzfile = np.load('C0_LGQ_data.npz')
f = npzfile['f'] #Model evaluations
m = npzfile['m'] #Number of inputs
n = npzfile['n'] #Dimension of active subspace
M = npzfile['M'] #Number of monte carlo samples
x0 = npzfile['x0'] #Sample data points
evals = npzfile['evals'] #Eigenvalues
W = npzfile['W'] #Eigenvectors
comp_flag = npzfile['comp_flag']
sub_br = npzfile['sub_br']
e_br = npzfile['e_br']
k = 1000 #number of data points used for plotting sufficient summary
# Rewrite the active/inactive subspace variables to be n-dimensional
W1 = W[:,:n]
W2 = W[:,n:]
# Define the active/inactive variables 
Y, Z = np.dot(x0[1:k,:], W1), np.dot(x0[1:k,:], W2)

Y = np.dot(x0[1:k,:], W1)
# Plot the active subspace info
if comp_flag == 0:
    #Plot MC estimated eigenvalues with bootstrap intervals
    active_subspaces.utils.plotters.eigenvalues(evals,e_br=e_br)
    #Plot bootstrap estimates of the subspace errors with confidence intervals
    active_subspaces.utils.plotters.subspace_errors(sub_br)
elif comp_flag == 1:
   active_subspaces.utils.plotters.eigenvalues(evals)
if (n <= 2):
    active_subspaces.utils.plotters.sufficient_summary(Y, f[1:k])