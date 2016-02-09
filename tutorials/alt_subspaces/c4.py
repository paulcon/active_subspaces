import numpy as np
import active_subspaces
import matplotlib.pyplot as plt
import test_function_3
# Set the number of parameter (m) 
m = 3
# Set the dimension of the active subspace (n)
n = 2
# Set the number of points used in the sufficient summary plot
k = 1000
# of Monte Carlo samples
M = 1000
# choose function
fun = test_function_3.fun

# generate an M-by-2m matrix of data points
x0 = 2*np.random.rand(M,2*m)-1
# partition data
x =  x0[:,:m]
y = x0[:,m:]

f = np.zeros(2*M)
f_x = np.zeros(M)
f_y = np.zeros(M)
c_index = 4 #C index number
comp_flag = 1 #0 for MC, 1 for LGW quadtrature

#function evaluations for f_x and f_y
for i in range(0,M):
    x_sample = x[i,:].reshape(m)
    [x_out, x_gradout] = fun(x_sample)
    f_x[i] = x_out
    
    y_sample = y[i,:].reshape(m)
    [y_out, y_gradout] = fun(y_sample)
    f_y[i] = y_out

f = np.append([[f_x]],[[f_y]],axis=0)
f = f.reshape(2*M)

sub = active_subspaces.subspaces.Subspaces()
### Call to compute C4 matrix
sub.compute(0 ,f ,x0,fun, c_index, comp_flag, 10, 200)

W = sub.eigenvectors
evals = sub.eigenvalues.reshape((m,1))

# Rewrite the active/inactive subspace variables to be n-dimensional
W1 = W[:,:n]
W2 = W[:,n:]
# Define the active/inactive variables 
x0 =  np.append([x],[y],axis=0).reshape((2*M,m))
#np.savez('C4_LGQ_data',m=m,n=n,M=M,f=f,x0=x0,evals=evals,W=W,comp_flag=comp_flag,sub_br=sub.sub_br,e_br=sub.e_br)


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

