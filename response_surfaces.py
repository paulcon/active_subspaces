import numpy as np
import zonotopes as zn
from asutils import VariableMap
from gaussian_quadrature import gauss_hermite
import pdb

def response_surface_design(W,n,N,NMC,bflag=0):
    
    # check len(N) == n
    m = W.shape[0]
    W1,W2 = W[:,:n],W[:,n:]
    
    if bflag:
        # uniform case
        if n==1:
            y0 = np.dot(W1.T,np.sign(W1))[0]
            if y0 < -y0:
                yl,yu = y0,-y0
                xl = np.sign(W1).reshape((1,m))
                xu = -np.sign(W1).reshape((1,m))
            else:
                yl,yu = -y0,y0
                xl = -np.sign(W1).reshape((1,m))
                xu = np.sign(W1).reshape((1,m))
            y = np.linspace(yl,yu,N[0]).reshape((N[0],1))
            
            Yzv = np.array([yl,yu])
            Xzv = np.vstack((xl,xu))
            Y = y[1:-1]
            
        else:
            Yzv,Xzv = zn.zonotope_vertices(W1)
            Y = zn.maximin_design(Yzv,N[0])
        
    else:
        # Gaussian case
        if len(N) != n:
            raise Exception('N should be a list of integers of length n.')
        Y = gauss_hermite(N)[0]
    
    vm = VariableMap(W,n,bflag)
    X,ind = vm.inverse(Y,NMC)
    if bflag:
        X = np.vstack((X,Xzv))
        ind = np.hstack((ind,np.arange(Y.shape[0],Y.shape[0]+Yzv.shape[0])))
        Y = np.vstack((Y,Yzv))
    
    return X,ind,Y
        
if __name__ == '__main__':
    m,n = 7,3
    N,NMC = [20],3
    bflag = 1
    W = np.linalg.qr(np.random.normal(size=(m,m)))[0]
    X,ind,Y = response_surface_design(W,n,N,NMC,bflag)
    print X
    print Y
    print ind
    if bflag:
        print np.linalg.norm(np.kron(Y[:N[0],:],np.ones((NMC,1))) - np.dot(X[:(N[0]*NMC),:],W[:,:n]))
    else:
        print np.linalg.norm(np.kron(Y,np.ones((NMC,1))) - np.dot(X,W[:,:n]))
