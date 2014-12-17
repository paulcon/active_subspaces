#!/usr/bin/python

import numpy as np
import pandas as pn
import active_subspaces as ac
import matplotlib.pyplot as plt

def quad_fun(x):
    A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
       [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
       [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
    f = 0.5*np.dot(x.T,np.dot(A,x))
    df = np.dot(A,x)
    return f,df

def sample_function(X,fun):
    M,m = X.shape
    F = np.zeros((M,1))
    dF = np.zeros((M,m))
    for i in range(M):
        x = X[i,:]
        f,df = fun(x.T)
        F[i] = f
        dF[i,:] = df.T
    return F,dF

def test_interface(fun):
    M,m = 30,3
    X = np.random.uniform(-1.0,1.0,(M,m))
    F,dF = sample_function(X,fun)
    
    # get active subspace with gradients from local linear grads
    lam,W,lam_br,sub_br = ac.get_active_subspace(dF,2)
    
    # make 1d sufficient summary plots
    y = np.dot(X,W[:,0])
    ac.sufficient_summary_plot(y,F,W[:,0])
    
    # make sufficient summary plots
    ac.plot_active_subspace(lam,W,lam_br=lam_br,sub_br=sub_br)

def test_load_data():

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
    
    # run linear model check
    # TODO: add check to make sure that there are enough rows in XX and F
    w,w_boot = ac.linear_model_check(XX,F[:,0])
     
    # sample from local linear gradients
    # TODO add check for number of rows
    XXX = np.random.uniform(-1.0,1.0,size=(100,m))
    G = ac.local_linear_gradients(XX,F[:,0],XXX)
    
    # get active subspace with gradients from local linear grads
    lam,W,lam_br,sub_br = ac.get_active_subspace(G,3)
    
    # make 1d sufficient summary plots
    y = np.dot(XX,W[:,:2])
    ac.sufficient_summary_plot(y,F[:,0],W[:,:2],w_boot=w_boot)
    
    # make sufficient summary plots
    ac.plot_active_subspace(lam,W,lam_br=lam_br,sub_br=sub_br)


if __name__ == "__main__":
    #test_load_data()
    test_interface(quad_fun)