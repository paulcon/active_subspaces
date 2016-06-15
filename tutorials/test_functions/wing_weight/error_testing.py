# checking derivatives with first-order finite differences
import wing_functions
reload(wing_functions)
from wing_functions import *
import numpy as np      
h = 1e-6;
m = 10
M = 5
x = np.random.uniform(-1, 1, (M, m))
f = wing(x)
grad_anal = wing_grad(x)
grad_fd = np.zeros((M,m))
for i in range(0,m):
    e = np.zeros((M, m))
    e[:,i] = 1
    step = wing(x+h*e)
    grad_fd[:,i] = np.squeeze((step-f)/h)

for k in range(M):
    print np.linalg.norm((grad_anal-grad_fd)[k])