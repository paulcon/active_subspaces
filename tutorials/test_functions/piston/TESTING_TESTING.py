import numpy as np
import active_subspaces
import matplotlib.pyplot as plt
import piston
# Set the number of parameter (m) 
m = 7
# Set the dimension of the active subspace (n)
n = 2
# Set the number of points used in the sufficient summary plot
k = 1000
# Compute ex0act solution for M randomly selected points
M = 100000
# Select function
function = piston.fun


x0 = 2*np.random.rand(M,m)-1
f = np.zeros(M)
df = np.zeros((M,m)) 
c_index = 4 #matrix number
comp_flag = 0 #0 for MC, 1 for LGW quadtrature
for i in range(0,M):
    sample = x0[i,:].reshape(m)
    [out, gradout] = function(sample)
    f[i] = out
    df[i,:] = gradout.T
    
sub = active_subspaces.subspaces.Subspaces()
sub.compute(df,f,x0,function, c_index,comp_flag,4,200)
W = sub.eigenvectors
evals = sub.eigenvalues.reshape((m,1))


# Rewrite the active/inactive subspace variables to be n-dimensional
W1 = W[:,:n]
W2 = W[:,n:]
# Define the active/inactive variables 
Y, Z = np.dot(x0[1:k,:], W1), np.dot(x0[1:k,:], W2)

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



#Gauss Legendre Quadrature of the C matrix
#xx = (np.ones(m)*k).astype(np.int64).tolist()  
#[x,w] = active_subspaces.utils.quadrature.gauss_legendre(xx)
#C = np.zeros((m,m))
#N = np.size(w)
#for i in range(0,N):
#    [Out,Gradout] = borehole.fun(x[i,:])
#    C = C + np.outer(Gradout,Gradout)*w[i]
## Eigenvalue decomposition    
#[evals,WW] = np.linalg.eig(C)
## Ordering eigenvalues in decending order
#order = np.argsort(evals)
#order = np.flipud(order)
#evals = evals[order]
#W = np.zeros((m,m))
#for jj in range(0,m):
#    W[:,jj] = WW[:,order[jj]]
#
#np.savez('data',m=m,n=n,M=M,f=f,df=df,x0=x0,evals=evals,W=W)