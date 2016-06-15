# checking derivatives with first-order finite differences
import piston_functions
reload(piston_functions)
from piston_functions import *
import numpy as np      
h = 1e-10;
m = 7;
M = 10
M0 = np.random.uniform(30, 60, (M, 1))
S = np.random.uniform(.005, .02, (M, 1))
V0 = np.random.uniform(.002, .01, (M, 1))
k = np.random.uniform(1000, 5000, (M, 1))
P0 = np.random.uniform(90000, 110000, (M, 1))
Ta = np.random.uniform(290, 296, (M, 1))
T0 = np.random.uniform(340, 360, (M, 1))
x = np.hstack((M0, S, V0, k, P0, Ta, T0))
f = piston(x)
grad_anal = piston_grad(x)
grad_fd = np.zeros((M,m))
for i in range(0,m):
    e = np.zeros((M, m))
    e[:,i] = 1
    step = piston(x+h*e)
    grad_fd[:,i] = np.squeeze((step-f)/h)

for k in range(M):
    print np.linalg.norm((grad_anal-grad_fd)[k])