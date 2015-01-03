import numpy as np
import zonotopes as zn
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def integration_rule(W,n,N,bflag=0):
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
            yc = 0.5*(y[1:]+y[:-1])
            
            NX = 1000
            X = np.random.uniform(-1.0,1.0,size=(NX,m))
            Y = np.dot(X,W1)
            w = np.histogram(Y,y)/float(NX)
            
            
        else:
            # get points
            yzv = zn.zonotope_vertices(W1)
            y = zn.maximin_design(yzv,N[0])
            T = Delaunay(y)
            c = []
            for t in T.simplices:
                c.append(np.mean(T.points[t],axis=0))
            yc = np.array(c)
            
            # approximate weights
            NX = 1000
            X = np.random.uniform(-1.0,1.0,size=(NX,m))
            Y = np.dot(X,W1)
            I = T.find_simplex(Y)
            w = np.zeros((T.nsimplex,1))
            for i in range(T.nsimplex):
                w[i] = np.sum(I==i)/float(NX)
            
    else:
        # Gaussian case
        print 'Gaussian not implemented'
        yc,w = None,None
        
    return yc,w
    
if __name__ == '__main__':
    m,n = 5,2
    W = np.linalg.qr(np.random.normal(size=(m,m)))[0]
    N = [10]
    y,w = integration_rule(W,n,N,bflag=1)
    
    plt.close('all')
    plt.figure()
    plt.scatter(y[:,0],y[:,1],s=60,c=w)
    plt.colorbar()
    plt.show()
    