import numpy as np
import zonotopes as zn
from gaussian_quadrature import gauss_hermite

def sample_z(N,y,W1,W2):
    mz = W2.shape[1]
    z0 = np.zeros((mz,1))
    s = np.dot(W1,y).reshape((W1.shape[0],1))
    
    # burn in
    for i in range(N):
        zc = z0 + 0.66*np.random.normal(size=z0.shape)
        if all(np.dot(W2,zc) <= 1-s) and all(np.dot(W2,zc) >= -1-s):
            z0 = zc
    
    # sample
    Z = np.zeros((mz,N))
    for i in range(N):
        zc = z0 + 0.66*np.random.normal(size=z0.shape)
        if all(np.dot(W2,zc) <= 1-s) and all(np.dot(W2,zc) >= -1-s):
            z0 = zc
        Z[:,i] = z0.reshape((z0.shape[0],))
        
    return Z

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
            else:
                yl,yu = -y0,y0
            y = np.linspace(yl,yu,N[0]).reshape((N[0],1))
        else:
            yzv = zn.zonotope_vertices(W1)
            y = zn.maximin_design(yzv,N[0])
        
        Ny = y.shape[0]
        
        # sample the z's 
        Zl = []
        for yp in y:
            Zl.append(sample_z(NMC,yp,W1,W2))
        Z = np.array(Zl).reshape((Ny,m-n,NMC))
        
    else:
        # Gaussian case
        y = gauss_hermite(N)[0]
        Ny = y.shape[0]
        
        # sample z's
        Z = np.random.normal(size=(Ny,m-n,NMC))
        
    # rotate back to x
    Y = np.tile(y.reshape((Ny,n,1)),(1,1,NMC))
    YZ = np.concatenate((Y,Z),axis=1).transpose((1,0,2)).reshape((m,NMC*Ny)).transpose((1,0))
    X = np.dot(YZ,W.T)
        
    return X,y
        
if __name__ == '__main__':
    X,y = response_surface_design(np.eye(3),2,[3],3,bflag=1)
    print X
    print y
