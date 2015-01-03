import numpy as np
import zonotopes as zn
import asutils as au
from gaussian_quadrature import gauss_hermite

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
            Yp = np.linspace(yl,yu,N[0]).reshape((N[0],1))
        else:
            yzv = zn.zonotope_vertices(W1)
            Yp = zn.maximin_design(yzv,N[0])
        
        # sample the z's 
        Ny = Yp.shape[0]
        Zlist = []
        for yp in Yp:
            Zlist.append(au.sample_z(NMC,yp,W1,W2))
        Z = np.array(Zlist).reshape((Ny,m-n,NMC))
        
    else:
        # Gaussian case
        Yp = gauss_hermite(N)[0]
        Ny = Yp.shape[0]
        
        # sample z's
        Z = np.random.normal(size=(Ny,m-n,NMC))
        
    # rotate back to x
    Y = np.tile(y.reshape((Ny,n,1)),(1,1,NMC))
    YZ = np.concatenate((Y,Z),axis=1).transpose((1,0,2)).reshape((m,NMC*Ny)).transpose((1,0))
    Xp = np.dot(YZ,W.T)
    
    # weights to get function values of x to function values of y
    Wp = (1.0/NMC)*np.kron(np.eye(Ny),np.ones(NMC))
    
    return Xp,Wp,Yp
        
if __name__ == '__main__':
    X,y = response_surface_design(np.eye(3),2,[3],3,bflag=1)
    print X
    print y
