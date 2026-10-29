import numpy as np     
import active_subspaces as ac 
def wing(xx):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns column vector of wing function at each row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    #Unnormalize inputs
    xl = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025])
    xu = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    Sw = x[:,0]; Wfw = x[:,1]; A = x[:,2]; L = x[:,3]*np.pi/180.; q = x[:,4]
    l = x[:,5]; tc = x[:,6]; Nz = x[:,7]; Wdg = x[:,8]; Wp = x[:,9]
    
    return (.036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp).reshape(M, 1)
    
def wing_grad(xx):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns matrix whose ith row is gradient of wing function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    
    #Unnormalize inputs
    xl = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025])
    xu = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    Sw = x[:,0]; Wfw = x[:,1]; A = x[:,2]; L = x[:,3]*np.pi/180.; q = x[:,4]
    l = x[:,5]; tc = x[:,6]; Nz = x[:,7]; Wdg = x[:,8]; Wp = x[:,9]
    
    Q = .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 #Convenience variable
    
    dfdSw = (.758*Q/Sw + Wp)[:,None]
    dfdWfw = (.0035*Q/Wfw)[:,None]
    dfdA = (.6*Q/A)[:,None]
    dfdL = (.9*Q*np.sin(L)/np.cos(L))[:,None]
    dfdq = (.006*Q/q)[:,None]
    dfdl = (.04*Q/l)[:,None]
    dfdtc = (-.3*Q/tc)[:,None]
    dfdNz = (.49*Q/Nz)[:,None]
    dfdWdg = (.49*Q/Wdg)[:,None]
    dfdWp = (Sw)[:,None]
    
    #The gradient components must be scaled in accordance with the chain rule: df/dx = df/dy*dy/dx
    return np.hstack((dfdSw*(200 - 150)/2., dfdWfw*(300 - 220)/2., dfdA*(10 - 6)/2., dfdL*(10 + 10)*np.pi/(2.*180), dfdq*(45 - 16)/2.,\
        dfdl*(1 - .5)/2., dfdtc*(.18 - .08)/2., dfdNz*(6 - 2.5)/2., dfdWdg*(2500 - 1700)/2., dfdWp*(.08 - .025)/2.))
    
    