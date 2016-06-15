import numpy as np
import active_subspaces as ac

def borehole(xx):
    #each row of xx should be [rw, r, Tu, Hu, Tl, Hl, L, Kw] in the normalized space
    #returns column vector of borehole function at each row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    #unnormalize inpus
    xl = np.array([63070, 990, 63.1, 700, 1120, 9855])
    xu = np.array([115600, 1110, 116, 820, 1680, 12045])
    x[:,2:] = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x[:,2:])
    x[:,0] = .0161812*x[:,0] + .1
    x[:,1] = np.exp(1.0056*x[:,1] + 7.71)   
    
    rw = x[:,0]; r = x[:,1]; Tu = x[:,2]; Hu = x[:,3]
    Tl = x[:,4]; Hl = x[:,5]; L = x[:,6]; Kw = x[:,7]    
    
    pi = np.pi
    
    return (2*pi*Tu*(Hu - Hl)/(np.log(r/rw)*(1 + 2*L*Tu/(np.log(r/rw)*rw**2*Kw) + Tu/Tl))).reshape(M, 1)
    
def borehole_grad(xx):
    #each row of xx should be [rw, r, Tu, Hu, Tl, Hl, L, Kw] in the normalized space
    #returns matrix whose ith row is gradient of borehole function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    #unnormalize inpus
    xl = np.array([63070, 990, 63.1, 700, 1120, 9855])
    xu = np.array([115600, 1110, 116, 820, 1680, 12045])
    x[:,2:] = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x[:,2:])
    x[:,0] = .0161812*x[:,0] + .1
    x[:,1] = np.exp(1.0056*x[:,1] + 7.71)   
    
    rw = x[:,0]; r = x[:,1]; Tu = x[:,2]; Hu = x[:,3]
    Tl = x[:,4]; Hl = x[:,5]; L = x[:,6]; Kw = x[:,7]
    
    pi = np.pi
    Q = 1 + 2*L*Tu/(np.log(r/rw)*rw**2*Kw) + Tu/Tl #Convenience variable
    l = np.log(r/rw) #Convenience variable
    
    dfdrw = (-2*pi*Tu*(Hu - Hl)*(Q*l)**-2*(-Q/rw - l*2*L*Tu/Kw*(l*rw**2)**-2*(-rw + 2*rw*l)))[:,None]
    dfdr = (-2*pi*Tu*(Hu - Hl)*(l*Q)**-2*(Q/r - 2*L*Tu/(r*rw**2*Kw*l)))[:,None]
    dfdTu = (2*pi*(Hu - Hl)/(l*Q) - 2*pi*Tu*(Hu - Hl)*(l*Q)**-2*(l*(2*L/(l*rw**2*Kw)+1./Tl)))[:,None]
    dfdHu = (2*pi*Tu/(l*Q))[:,None]
    dfdTl = (2*pi*Tu*(Hu - Hl)*(l*Q)**-2*l*Tu/Tl**2)[:,None]
    dfdHl = (-2*pi*Tu/(l*Q))[:,None]
    dfdL = (-2*pi*Tu*(Hu - Hl)*(l*Q)**-2*2*Tu/(rw**2*Kw))[:,None]
    dfdKw = (2*pi*Tu*(Hu - Hl)*(l*Q)**-2*2*L*Tu/(rw**2*Kw**2))[:,None]
    
    #The gradient components must be scaled in accordance with the chain rule: df/dx = df/dy*dy/dx
    r = np.log(r); r = ((r - 7.71)/1.0056).reshape(M, 1)
    return np.hstack((dfdrw*.0161812, dfdr*1.0056*np.exp(1.0056*r + 7.71), dfdTu*(115600 - 63070)/2., dfdHu*(1110 - 990)/2.,\
        dfdTl*(116 - 63.1)/2., dfdHl*(820 - 700)/2., dfdL*(1680 - 1120)/2., dfdKw*(12045 - 9855)/2.))
    