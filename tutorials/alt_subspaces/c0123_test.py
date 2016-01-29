import numpy as np
import active_subspaces
import matplotlib.pyplot as plt
import borehole
# Set the number of parameter (m) 
m = 8
# Set the dimension of the active subspace (n)
n = 2
# Set the number of points used in the sufficient summary plot
k = 1000
# Compute ex0act solution for M randomly selected points
M = 1000

# Select function
#For c_index =  0,1,2,3
x0 = 2*np.random.rand(M,m)-1
f = np.zeros(M)
df = np.zeros((M,m)) 
c_index = 0 #matrix number
comp_flag = 1 #0 for MC, 1 for LGW quadtrature
for i in range(0,M):
    sample = x0[i,:].reshape(m)
    [out, gradout] = borehole.fun(sample)
    f[i] = out
    df[i,:] = gradout.T
    
    
sub = active_subspaces.subspaces.Subspaces()
sub.compute(df ,f ,x0,borehole.fun, c_index, comp_flag, 5, 200)
#sub.compute(df)
W = sub.eigenvectors
evals = sub.eigenvalues.reshape((m,1))
np.savez('C0_LGQ_data',m=m,n=n,M=M,f=f,x0=x0,evals=evals,W=W,comp_flag=comp_flag,sub_br=sub.sub_br,e_br=sub.e_br)


# Rewrite the active/inactive subspace variables to be n-dimensional
W1 = W[:,:n]
W2 = W[:,n:]
# Define the active/inactive variables 
Y = np.dot(x0[1:k,:], W1)

# Plot the active subspace info
if comp_flag == 0:
    #Plot MC estimated eigenvalues with bootstrap intervals
    active_subspaces.utils.plotters.eigenvalues(sub.eigenvalues,e_br=sub.e_br)
    #Plot bootstrap estimates of the subspace errors with confidence intervals
    active_subspaces.utils.plotters.subspace_errors(sub.sub_br)
elif comp_flag == 1:
   active_subspaces.utils.plotters.eigenvalues(sub.eigenvalues)
if (n <= 2):
    active_subspaces.utils.plotters.sufficient_summary(Y, f[1:k])
