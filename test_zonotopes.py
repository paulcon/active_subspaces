import numpy as np
import zonotopes as zn
import matplotlib.pyplot as plt
import scipy.spatial as sp

if __name__ == '__main__':
    m,n = 10,2
    W1 = np.linalg.qr(np.random.normal(size=(m,n)))[0]
    yzv = zn.zonotope_vertices(W1)
    N = 20
    Y,res = zn.maximin_design(yzv,N)
    
    plt.close('all')
    plt.figure()
    plt.plot(Y[:,0],Y[:,1],'k.')
    plt.axes().set_aspect('equal')
    
    V = sp.Voronoi(Y)
    sp.voronoi_plot_2d(V)
    plt.axes().set_aspect('equal')
        
    T = sp.Delaunay(Y)
    C = []
    for t in T.simplices:
        C.append(np.mean(T.points[t],axis=0))
    centroids = np.array(C)
    sp.delaunay_plot_2d(T)
    plt.plot(centroids[:,0],centroids[:,1],'ro')
    plt.axes().set_aspect('equal')
        
    plt.show()