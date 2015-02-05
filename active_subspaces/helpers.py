import numpy as np

def quadratic_model_check(X,f,gamma,k):
    M,m = X.shape
    gamma = gamma.reshape((1,m))
    
    pr = rg.PolynomialRegression(2)
    pr.train(X,f)

    # get regression coefficients
    b,A = pr.g,pr.H
    
    # compute eigenpairs
    e,W = np.linalg.eig(np.outer(b,b.T) + \
        np.dot(A,np.dot(np.diagflat(gamma),A)))
    ind = np.argsort(e)[::-1]
    e,W = e[ind],W[:,ind]*np.sign(W[0,ind])
    
    return e[:k],W

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
        Y = gq.gauss_hermite(N)[0]
    
    vm = VariableMap(W,n,bflag)
    X,ind = vm.inverse(Y,NMC)
    if bflag:
        X = np.vstack((X,Xzv))
        ind = np.hstack((ind,np.arange(Y.shape[0],Y.shape[0]+Yzv.shape[0])))
        Y = np.vstack((Y,Yzv))
    
    return X,ind,Y

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
            Wy.reshape((N[0]-1,1))
            
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
    X,ind = vm.inverse(Y,NMC)
    
    return X,ind,Y,Wy
    