import numpy as np     
import active_subspaces as ac 
def robot(xx):
    #each row of xx should be [t1, t2, t3, t4, L1, L2, L3, L4] in the normalized input space
    #returns a column vector of the piston function at each row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    #Unnormalize inputs
    xl = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    xu = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1, 1, 1, 1])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    t1 = x[:,0]; t2 = x[:,1]; t3 = x[:,2]; t4 = x[:,3]
    L1 = x[:,4]; L2 = x[:,5]; L3 = x[:,6]; L4 = x[:,7]
    
    cos = np.cos
    sin = np.sin
    
    u = L1*cos(t1) + L2*cos(t1 + t2) + L3*cos(t1 + t2 + t3) + L4*cos(t1 + t2 + t3 + t4)
    v = L1*sin(t1) + L2*sin(t1 + t2) + L3*sin(t1 + t2 + t3) + L4*sin(t1 + t2 + t3 + t4)
    
    return ((u**2 + v**2)**.5).reshape(M, 1)
    
def robot_grad(xx):
    #each row of xx should be [t1, t2, t3, t4, L1, L2, L3, L4] in the normalized input space
    #returns a matrix whose ith row is gradient of robot function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    #Unnormalize inputs
    xl = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    xu = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1, 1, 1, 1])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    t1 = x[:,0]; t2 = x[:,1]; t3 = x[:,2]; t4 = x[:,3]
    L1 = x[:,4]; L2 = x[:,5]; L3 = x[:,6]; L4 = x[:,7]
    
    cos = np.cos
    sin = np.sin
    
    u = L1*cos(t1) + L2*cos(t1 + t2) + L3*cos(t1 + t2 + t3) + L4*cos(t1 + t2 + t3 + t4)
    v = L1*sin(t1) + L2*sin(t1 + t2) + L3*sin(t1 + t2 + t3) + L4*sin(t1 + t2 + t3 + t4)
        
    dfdt1 = np.zeros((M, 1))
    dfdt2 = -((u**2 + v**2)**-.5*(L1*(v*cos(t1) - u*sin(t1))))[:,None]
    dfdt3 = -((u**2 + v**2)**-.5*(L1*(v*cos(t1) - u*sin(t1)) + L2*(v*cos(t1 + t2) - u*sin(t1 + t2))))[:,None]
    dfdt4 = -((u**2 + v**2)**-.5*(L1*(v*cos(t1) - u*sin(t1)) + L2*(v*cos(t1 + t2) - u*sin(t1 + t2)) + \
        L3*(v*cos(t1 + t2 + t3) - u*sin(t1 + t2 + t3))))[:,None]
    dfdL1 = (.5*(u**2 + v**2)**-.5*(2*u*cos(t1) + 2*v*sin(t1)))[:,None]
    dfdL2 = (.5*(u**2 + v**2)**-.5*(2*u*cos(t1 + t2) + 2*v*sin(t1 + t2)))[:,None]
    dfdL3 = (.5*(u**2 + v**2)**-.5*(2*u*cos(t1 + t2 + t3) + 2*v*sin(t1 + t2 + t3)))[:,None]
    dfdL4 = (.5*(u**2 + v**2)**-.5*(2*u*cos(t1 + t2 + t3 + t4) + 2*v*sin(t1 + t2 + t3 + t4)))[:,None]
    
    #The gradient components must be scaled in accordance with the chain rule: df/dx = df/dy*dy/dx
    return np.hstack((dfdt1*(2*np.pi)/2., dfdt2*(2*np.pi)/2., dfdt3*(2*np.pi)/2., dfdt4*(2*np.pi)/2., dfdL1*(1)/2.,\
        dfdL2*(1)/2., dfdL3*(1)/2., dfdL4*(1)/2.))
    
    