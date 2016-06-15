# checking derivatives with first-order finite differences
from otlcircuit_functions import *
import numpy as np      
h = 1e-4;
m = 6;
M = 10
Rb1 = np.random.uniform(50, 150, (M,1))
Rb2 = np.random.uniform(25, 70, (M,1))
Rf = np.random.uniform(.5, 3, (M,1))
Rc1 = np.random.uniform(1.2, 2.5, (M,1))
Rc2 = np.random.uniform(.25, 1.2, (M,1))
beta = np.random.uniform(50, 300, (M,1))
x = np.hstack((Rb1, Rb2, Rf, Rc1, Rc2, beta))
f = circuit(x)
grad_anal = circuit_grad(x)
grad_fd = np.zeros((M,m))
for i in range(0,m):
    e = np.zeros((M, m))
    e[:,i] = 1
    step = circuit(x+h*e)
    grad_fd[:,i] = np.squeeze((step-f)/h)

for k in range(M):
    print np.linalg.norm((grad_anal-grad_fd)[k])