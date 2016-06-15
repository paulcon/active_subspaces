# checking derivatives with first-order finite differences
import robot_functions
reload(robot_functions)
from robot_functions import *
import numpy as np      
h = 1e-8
m = 8;
M = 10
theta = np.random.uniform(-1, 1, (M, 4))
L = np.random.uniform(-1, 1, (M, 4))
x = np.hstack((theta, L))
f = robot(x)
grad_anal = robot_grad(x)
grad_fd = np.zeros((M,m))
for i in range(0,m):
    e = np.zeros((M, m))
    e[:,i] = 1
    step = robot(x+h*e)
    grad_fd[:,i] = np.squeeze((step-f)/h)

for k in range(M):
    print np.linalg.norm((grad_anal-grad_fd)[k])