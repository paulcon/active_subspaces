#!/usr/bin/python

import numpy as np
import pandas as pn
import active_subspaces as ac
import matplotlib.pyplot as plt

# load data
df = pn.DataFrame.from_csv('HyShotII.txt')
data = df.values
X = data[:,0:7]
F = data[:,7:9]
M = F.shape[0]
m = X.shape[1]

# normalize inputs
xl = np.array([2.6,0.1,122.5,1.6448e7,3.0551e6,0.6*0.05,0.6*0.145])
xu = np.array([4.6,1.9,367.5,1.9012e7,3.4280e6,1.4*0.05,1.4*0.145])
XX = ac.normalize_uniform(X,xl,xu)

#w0 = ac.lingrad(XX,F[:,0])
w,w_boot = ac.linear_model_check(XX,F[:,0])
#    
#
#XXX = np.random.uniform(-1.0,1.0,size=(100,m))
#G = ac.local_linear_gradients(XX,F[:,0],XXX)
#
#lam,W,lam_bootrange,sub_bootrange = ac.get_active_subspace(G,3)
#
#y = np.dot(XX,W[:,:2])
##ac.sufficient_summary_plot(y,F[:,0],W[:,:2],w_boot=w_boot)
#
## testing quadratic model
##ZZ = np.random.uniform(-1.0,1.0,size=(100,7))
##ff,ee,QQ,Atrue = ac.quadtest(ZZ)
##
#gamma = (1.0/3.0)*np.ones((m,1))
#lam2,W2,lam_bootrange2,sub_bootrange2 = ac.quadratic_model_check(XX,F[:,0],gamma,3)
#
#ac.plot_active_subspace(lam,W,
#    lam_bootrange=lam_bootrange,sub_bootrange=sub_bootrange)


