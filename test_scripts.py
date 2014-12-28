#!/usr/bin/python

import numpy as np
import pandas as pn
import active_subspaces as ac
import matplotlib.pyplot as plt

def quadtest(X):
    M,m = X.shape
    B = np.random.normal(size=(m,m))
    Q = np.linalg.qr(B)[0]
    e = np.array([10**(-i) for i in range(1,m+1)])
    A = np.dot(Q,np.dot(np.diagflat(e),Q.T))
    f = np.zeros((M,1))
    for i in range(M):
        z = X[i,:]
        f[i] = 0.5*np.dot(z,np.dot(A,z.T))
        
    return f,e,Q,A

def quad_fun(x):
    A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
       [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
       [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
    f = 0.5*np.dot(x.T,np.dot(A,x))
    df = np.dot(A,x)
    return f,df

def quad_fun_nograd(x):
    A = np.array([[ 0.2406659045776698, -0.3159904335007421, -0.1746908591702878],
       [-0.3159904335007421,  0.5532215729009683,  0.3777995408101305],
       [-0.1746908591702878,  0.3777995408101305,  0.3161125225213613]])
    f = 0.5*np.dot(x.T,np.dot(A,x))
    return f
        
def test_interface(fun):
    M,m = 300,3
    X = np.random.uniform(-1.0,1.0,(M,m))
    F,dF = ac.sample_function(X,fun,dflag=True)
    
    e,W,e_br,sub_br = ac.compute_active_subspace(dF,2)
    
    ac.plot_eigenvalues(e,e_br=e_br)
    ac.plot_subspace_errors(sub_br)
    ac.plot_eigenvectors(W[:,:1])
    
    # make 1d sufficient summary plots
    y = np.dot(X,W[:,0])
    ac.sufficient_summary_plot(y,F)
    
    # make 2d sufficient summary plots
    y = np.dot(X,W[:,:2])
    ac.sufficient_summary_plot(y,F)    
    
    return 0
    
def test_interface_nograd(fun):
    MM,M,m = 30,300,3
    X = np.random.uniform(-1.0,1.0,(M,m))
    F = ac.sample_function(X,fun,dflag=False)

    # test finite difference gradients
    ddF = ac.finite_difference_gradients(X,fun)
    
    # test local linear model gradients
    XX = np.random.uniform(-1.0,1.0,(MM,m))
    ddF = ac.local_linear_gradients(X,F,XX)
    
    e,W,e_br,sub_br = ac.compute_active_subspace(ddF,2)
    
    ac.plot_eigenvalues(e,e_br=e_br)
    ac.plot_subspace_errors(sub_br)
    ac.plot_eigenvectors(W[:,:1])
    
    # make 1d sufficient summary plots
    y = np.dot(X,W[:,0])
    ac.sufficient_summary_plot(y,F)

    # make 2d sufficient summary plots
    y = np.dot(X,W[:,:2])
    ac.sufficient_summary_plot(y,F)    

    return 0

def test_load_data():

    # load data
    df = pn.DataFrame.from_csv('quad_nograd.txt')
    data = df.values
    X = data[:,0:3]
    F = data[:,3]
    M,m = X.shape
    
    # test local linear model gradients
    MM = 100
    XX = np.random.uniform(-1.0,1.0,(MM,m))
    ddF = ac.local_linear_gradients(X,F,XX)
    
    e,W,e_br,sub_br = ac.compute_active_subspace(ddF,2)
    
    ac.plot_eigenvalues(e,e_br=e_br)
    ac.plot_subspace_errors(sub_br)
    ac.plot_eigenvectors(W[:,:1])
    
    # make 1d sufficient summary plots
    y = np.dot(X,W[:,0])
    ac.sufficient_summary_plot(y,F)

    # make 2d sufficient summary plots
    y = np.dot(X,W[:,:2])
    ac.sufficient_summary_plot(y,F) 
    
    # load data
    df = pn.DataFrame.from_csv('quad.txt')
    data = df.values
    X = data[:,:3]
    F = data[:,3]
    dF = data[:,4:]
    M,m = X.shape
    
    e,W,e_br,sub_br = ac.compute_active_subspace(dF,2)
    
    ac.plot_eigenvalues(e,e_br=e_br)
    ac.plot_subspace_errors(sub_br)
    ac.plot_eigenvectors(W[:,:1])
    
    # make 1d sufficient summary plots
    y = np.dot(X,W[:,0])
    ac.sufficient_summary_plot(y,F)

    # make 2d sufficient summary plots
    y = np.dot(X,W[:,:2])
    ac.sufficient_summary_plot(y,F) 
    
    return 0

def test_quick_check():
    # load data
    df = pn.DataFrame.from_csv('quad_nograd.txt')
    data = df.values
    X = data[:,0:3]
    F = data[:,3]
    w = ac.quick_check(X,F)
    return 0

def write_quad_csv():
    X = np.random.uniform(-1.0,1.0,(100,3))
    D = np.zeros((100,4))
    for i in range(100):
        x = X[i,:]
        f = quad_fun_nograd(x.T)
        D[i,:3] = x.T
        D[i,3] = f
        
    labels = ['p1','p2','p3','output']
    df = pn.DataFrame(data=D,columns=labels)
    df.to_csv('quad_nograd.txt')
    
    X = np.random.uniform(-1.0,1.0,(100,3))
    D = np.zeros((100,7))
    for i in range(100):
        x = X[i,:]
        f,df = quad_fun(x.T)
        D[i,:3] = x.T
        D[i,3] = f
        D[i,4:] = df.T
        
    labels = ['p1','p2','p3','output','d1','d2','d3']
    df = pn.DataFrame(data=D,columns=labels)
    df.to_csv('quad.txt')


if __name__ == "__main__":
    
    #if not test_load_data():
    #    print 'Success!'
    
    #if not test_interface(quad_fun):
    #    print 'Success!'
    
    #if not test_interface_nograd(quad_fun_nograd):
    #    print 'Success!'
    
    if not test_quick_check():
        print 'Success!'
    

    