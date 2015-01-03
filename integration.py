import numpy as np
import zonotopes as zn
import gaussian_quadrature as gq
import asutils as au
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pdb

def integration_rule(W,n,N,NMC,bflag=0):
    m = W.shape[0]
    W1,W2 = W[:,:n],W[:,n:]
    NX = 10000
    
    if bflag:
        # uniform case
        if n==1:
            y0 = np.dot(W1.T,np.sign(W1))[0]
            if y0 < -y0:
                yl,yu = y0,-y0
            else:
                yl,yu = -y0,y0
            y = np.linspace(yl,yu,N[0]).reshape((N[0],1))
            Yp = 0.5*(y[1:]+y[:-1])
            
            Ysamp = np.dot(np.random.uniform(-1.0,1.0,size=(NX,m)),W1)
            #pdb.set_trace()
            Wy = np.histogram(Ysamp.reshape((NX,)),bins=y.reshape((N[0],)), \
                range=(np.amin(y),np.amax(y)))[0]/float(NX)
            
        else:
            # get points
            yzv = zn.zonotope_vertices(W1)
            y = zn.maximin_design(yzv,N[0])
            T = Delaunay(y)
            c = []
            for t in T.simplices:
                c.append(np.mean(T.points[t],axis=0))
            Yp = np.array(c)
            
            # approximate weights
            Ysamp = np.dot(np.random.uniform(-1.0,1.0,size=(NX,m)),W1)
            I = T.find_simplex(Ysamp)
            Wy = np.zeros((T.nsimplex,1))
            for i in range(T.nsimplex):
                Wy[i] = np.sum(I==i)/float(NX)
                
        # sample the z's
        Ny = Yp.shape[0]
        Zlist = []
        for yp in Yp:
            Zlist.append(au.sample_z(NMC,yp,W1,W2))
        Z = np.array(Zlist).reshape((Ny,m-n,NMC))
            
    else:
        # Gaussian case
        Yp,Wy = gq.gauss_hermite(N)
        # check that Yp.shape[1] = n
        Ny = Yp.shape[0]
        
        # sample z's
        Z = np.random.normal(size=(Ny,m-n,NMC))
        
    # rotate back to x
    pdb.set_trace()
    Y = np.tile(Yp.reshape((Ny,n,1)),(1,1,NMC))
    YZ = np.concatenate((Y,Z),axis=1).transpose((1,0,2)).reshape((m,NMC*Ny)).transpose((1,0))
    Xp = np.dot(YZ,W.T)
    
    # weights for integration in x space
    Wx = np.kron(Wy,(1.0/NMC)*np.ones((NMC,1)))
    
    return Xp,Wx,Yp,Wy
    
if __name__ == '__main__':
    m,n = 5,2
    W = np.linalg.qr(np.random.normal(size=(m,m)))[0]
    N = [15,15]
    NMC = 10
    Xp,Wx,Yp,Wy = integration_rule(W,n,N,NMC,bflag=0)
    
    plt.close('all')
    plt.figure()
    if n==2:
        plt.scatter(Yp[:,0],Yp[:,1],s=60,c=Wy)
        plt.colorbar()
    elif n==1:
        plt.plot(Yp,Wy,'ro')
    plt.show()
    