import numpy as np
import active_subspaces
import matplotlib.pyplot as plt
import borehole
# Set the number of parameter (m) 
m = 8
# Set the dimension of the active subspace (n)
n = 2
# Set the number of points per dimension for Gauss Legendre Quadrature
k = 6
# Compute ex0act solution for M randomly selected points
M = 100000
x0 = 2*np.random.rand(M,m)-1
f = np.zeros(M)
df = np.zeros((M,m))
for i in range(0,M):
    sample = x0[i,:].reshape(m)
    [out, gradout] = borehole.fun(sample)
    f[i] = out
    df[i,:] = gradout.T
    
#Gauss Legendre Quadrature of the C matrix
xx = (np.ones(m)*k).astype(np.int64).tolist()  
[x,w] = active_subspaces.utils.quadrature.gauss_legendre(xx)
C = np.zeros((m,m))
N = np.size(w)
for i in range(0,N):
    [Out,Gradout] = borehole.fun(x[i,:])
    C = C + np.outer(Gradout,Gradout)*w[i]
# Eigenvalue decomposition    
[evals,WW] = np.linalg.eig(C)
# Ordering eigenvalues in decending order
order = np.argsort(evals)
order = np.flipud(order)
evals = evals[order]
W = np.zeros((m,m))
for jj in range(0,m):
    W[:,jj] = WW[:,order[jj]]

np.savez('data',m=m,n=n,M=M,f=f,df=df,x0=x0,evals=evals,W=W)