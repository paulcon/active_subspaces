import numpy as np
import active_subspaces
import matplotlib.pyplot as plt

# Load the data 
npzfile = np.load('data.npz')
f = npzfile['f'] #Model evaluations
df = npzfile['df'] #Gradient evaluations
m = npzfile['m'] #Number of inputs
n = npzfile['n'] #Dimension of active subspace
M = npzfile['M'] #Number of monte carlo samples
x0 = npzfile['x0'] #Sample data points
evals = npzfile['evals'] #Eigenvalues
W = npzfile['W'] #Eigenvectors
k = 1000 #number of data points used for plotting sufficient summary

sub = active_subspaces.subspaces.Subspaces()
sub.compute(df, n_boot=100)

# Rewrite the active/inactive subspace variables to be n-dimensional
W1 = W[:,:n]
W2 = W[:,n:]
# Define the active/inactive variables 
Y, Z = np.dot(x0[1:k,:], W1), np.dot(x0[1:k,:], W2)

# Plot the active subspace info
#Plot MC estimated eigenvalues with bootstrap intervals
active_subspaces.utils.plotters.eigenvalues(sub.eigenvalues,e_br=sub.e_br)
#Plot Quadrature eigenvalues 
active_subspaces.utils.plotters.eigenvalues(evals)
#Plot bootstrap estimates of the subspace errors with confidence intervals
active_subspaces.utils.plotters.subspace_errors(sub.sub_br)
if (n <= 2):
    active_subspaces.utils.plotters.sufficient_summary(Y, f[1:k])