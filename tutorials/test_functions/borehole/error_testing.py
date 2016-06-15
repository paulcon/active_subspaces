# checking derivatives with first-order finite differences
import borehole_functions as bh
import numpy as np      
h = 1e-6;
m = 8;
M = 10
rw = np.random.normal(.1, .0161812, (M, 1))
r = np.exp(np.random.normal(7.71, 1.0056, (M, 1)))
Tu = np.random.uniform(63070, 115600, (M,1))
Hu = np.random.uniform(990, 1110, (M,1))
Tl = np.random.uniform(63.1, 116, (M,1))
Hl = np.random.uniform(700, 820, (M,1))
L = np.random.uniform(1120, 1680, (M,1))
Kw = np.random.uniform(9855, 12045, (M,1))
x = np.hstack((rw, r, Tu, Hu, Tl, Hl, L, Kw))
f = bh.borehole(x)
bh.borehole_grad(x)
grad_anal = bh.borehole_grad(x)
print grad_anal
grad_fd = np.zeros((M,m))
for i in range(0,m):
    e = np.zeros((M, m))
    e[:,i] = 1
    step = bh.borehole(x+h*e)
    grad_fd[:,i] = np.squeeze((step-f)/h)

for k in range(M):
    print np.linalg.norm((grad_anal-grad_fd)[k])