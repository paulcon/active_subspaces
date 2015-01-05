import numpy as np
import zonotopes as zn
import gaussian_quadrature as gq
from asutils import VariableMap
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

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
            Y = 0.5*(y[1:]+y[:-1])
            
            Ysamp = np.dot(np.random.uniform(-1.0,1.0,size=(NX,m)),W1)
            Wy = np.histogram(Ysamp.reshape((NX,)),bins=y.reshape((N[0],)), \
                range=(np.amin(y),np.amax(y)))[0]/float(NX)
            
        else:
            # get points
            yzv = zn.zonotope_vertices(W1)[0]
            y = np.vstack((yzv,zn.maximin_design(yzv,N[0])))
            T = Delaunay(y)
            c = []
            for t in T.simplices:
                c.append(np.mean(T.points[t],axis=0))
            Y = np.array(c)
            
            # approximate weights
            Ysamp = np.dot(np.random.uniform(-1.0,1.0,size=(NX,m)),W1)
            I = T.find_simplex(Ysamp)
            Wy = np.zeros((T.nsimplex,1))
            for i in range(T.nsimplex):
                Wy[i] = np.sum(I==i)/float(NX)
            
    else:
        # Gaussian case
        if len(N) != n:
            raise Exception('N should be a list of integers of length n.')
        Y,Wy = gq.gauss_hermite(N)
        
    vm = VariableMap(W,n,bflag)
    X = vm.inverse(Y,NMC)
    
    # weights for integration in x space
    Wx = np.kron(Wy,(1.0/NMC)*np.ones((NMC,1)))
    
    return X,Wx,Y,Wy
    
if __name__ == '__main__':
    m,n = 5,1
    W = np.linalg.qr(np.random.normal(size=(m,m)))[0]
    N = [15]
    NMC = 10
    X,Wx,Y,Wy = integration_rule(W,n,N,NMC,bflag=0)
    
    plt.close('all')
    plt.figure()
    if n==2:
        plt.scatter(Y[:,0],Y[:,1],s=60,c=Wy)
        plt.colorbar()
    elif n==1:
        plt.plot(Y,Wy,'ro')
    plt.show()
    