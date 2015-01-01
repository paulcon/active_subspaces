import numpy as np
from scipy.spatial import ConvexHull,distance_matrix
from scipy.optimize import minimize
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Mesh_2 import Mesh_2_Constrained_Delaunay_triangulation_2
from CGAL.CGAL_Mesh_2 import Delaunay_mesh_size_criteria_2
from CGAL import CGAL_Mesh_2

def nzv(m,n,M=None):
    # number of zonotope vertices
    if M is None:
        M = np.zeros((m,n))
    if m==1 or n==1:
        M[m-1,n-1] = 2
    elif M[m-1,n-1]==0:
        k1,M = nzv(m-1,n-1,M)
        k2,M = nzv(m-1,n,M)
        M[m-1,n-1] = k1 + k2
        for i in range(n-1):
            M = nzv(m,i+1,M)[1]
    k = M[m-1,n-1]
    return k,M

def zonotope_vertices(W1):
    m,n = W1.shape
    NMC = 10000
    Xlist = []
    for i in range(NMC):
        y = np.random.normal(size=(n))
        x = np.sign(np.dot(y,W1.transpose()))
        addx = True
        for xx in Xlist:
            if all(x==xx):
                addx = False
                break
        if addx:
            Xlist.append(x)
    X = np.array(Xlist)
    Y = np.dot(X,W1)
    return Y
    
def maximin_design_obj(y,yzv=None):
    Ny,n = yzv.shape
    N = y.size/n
    Y = np.vstack((y.reshape((N,n)),yzv))
    Dy = distance_matrix(Y,Y) + 1e4*np.eye(N+Ny)
    return -np.amin(Dy.flatten())

def maximin_design(yzv,N):
    n = yzv.shape[1]
    C = ConvexHull(yzv)
    A = np.kron(np.eye(N),C.equations[:,:n])
    b = np.kron(np.ones(N),C.equations[:,n])
    cons = ({'type':'ineq',
                'fun' : lambda x: np.dot(A,x)-b,
                'jac' : lambda x: A})
    y0 = 0.1*np.random.normal(size=(N,n))
    res = minimize(maximin_design_obj,y0,args=(yzv,),constraints=cons,
                    method='SLSQP',options={'disp':True,'maxiter':1e9,'ftol':1e-12})
    Y = np.vstack((yzv,res.x.reshape(y0.shape)))
    return Y
    
def mesh_zonotope_2(W1):
    # check that W1.shape[1] == 2
    ymsh = Mesh_2_Constrained_Delaunay_triangulation_2()
    
    Yz = zonotope_vertices(W1)
    Yhull = ConvexHull(Yz)
    for s in Yhull.simplices:
        p0 = ymsh.insert(Point_2(Yz[s[0],0],Yz[s[0],1]))
        p1 = ymsh.insert(Point_2(Yz[s[1],0],Yz[s[1],1]))
        ymsh.insert_constraint(p0,p1)
        
    print "Number of vertices: ", ymsh.number_of_vertices()
    print "Meshing the triangulation..."
    CGAL_Mesh_2.refine_Delaunay_mesh_2(ymsh,Delaunay_mesh_size_criteria_2(0.125, 1.0))    
    print "Number of vertices: ", ymsh.number_of_vertices()
    
    Yl = []
    for p in ymsh.points():
        Yl.append(np.array([p.x(),p.y()]))
    return np.array(Yl)
    
if __name__ == '__main__':
    m,n = 20,4
    W1 = np.linalg.qr(np.random.normal(size=(m,n)))[0]
    yzv = zonotope_vertices(W1)
    N = 20
    Y,res = maximin_design(yzv,N)

    
    
