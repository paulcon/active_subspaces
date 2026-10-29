import numpy as np     
import active_subspaces as ac 
def circuit(xx):
    #each row of xx should be [Rb1, Rb2, Rf, Rc1, Rc2, beta] in the normalized input space
    #returns column vector of circuit function at each row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M = x.shape[0]
    
    #Unnormalize inputs
    xl = np.array([50, 25, .5, 1.2, .25, 50])
    xu = np.array([150, 70, 3, 2.5, 1.2, 300])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    Rb1 = x[:,0]; Rb2 = x[:,1]; Rf = x[:,2]
    Rc1 = x[:,3]; Rc2 = x[:,4]; beta = x[:,5]
    
    Vb1 = 12*Rb2/(Rb1 + Rb2)
    denom = beta*(Rc2 + 9) + Rf #Convenience variable
    
    return ((Vb1 + .74)*beta*(Rc2 + 9)/denom + 11.35*Rf/denom + .74*Rf*beta*(Rc2 + 9)/(Rc1*denom)).reshape(M, 1)
    
def circuit_grad(xx):
    #each row of xx should be [Rb1, Rb2, Rf, Rc1, Rc2, beta] in the normalized input space
    #returns matrix whose ith row is gradient of circuit function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    
    #Unnormalize inputs
    xl = np.array([50, 25, .5, 1.2, .25, 50])
    xu = np.array([150, 70, 3, 2.5, 1.2, 300])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    Rb1 = x[:,0]; Rb2 = x[:,1]; Rf = x[:,2]
    Rc1 = x[:,3]; Rc2 = x[:,4]; beta = x[:,5]
    
    Vb1 = 12*Rb2/(Rb1 + Rb2)
    denom = beta*(Rc2 + 9) + Rf #Convenience variable
    
    dvdRb1 = (-12.*Rb2*beta*(Rc2 + 9.)/(denom*(Rb1 + Rb2)**2))[:,None]
    dvdRb2 = (beta*(Rc2 + 9.)/denom*(12./(Rb1 + Rb2) - 12.*Rb2/(Rb1 + Rb2)**2))[:,None]
    dvdRf = (-(Vb1 + .74)*beta*(Rc2 + 9.)/denom**2 + 11.35/denom - 11.35*Rf/denom**2 + .74*beta*(Rc2 + 9.)/(Rc1*denom)-\
        .74*Rf*beta*(Rc2 + 9.)/(Rc1*denom**2))[:,None]
    dvdRc1 = (-.74*Rf*beta*(Rc2 + 9.)/(Rc1**2*denom))[:,None]
    dvdRc2 = (beta*(Vb1+.74)/denom - (Rc2 + 9.)*beta**2*(Vb1 + .74)/denom**2 - 11.35*Rf*beta/denom**2 +\
        .74*Rf*beta/(Rc1*denom) - .74*Rf*beta**2*(Rc2 + 9)/(Rc1*denom**2))[:,None]
    dvdbeta = ((Vb1 + .74)*(Rc2 + 9.)/denom - (Vb1 + .74)*beta*(Rc2 + 9.)**2/denom**2 - 11.35*Rf*(Rc2 + 9.)/denom**2 +\
        .74*Rf*(Rc2 + 9.)/(Rc1*denom) - .74*Rf*beta*(Rc2 + 9.)**2/(Rc1*denom**2))[:,None]
    
    #The gradient components must be scaled in accordance with the chain rule: df/dx = df/dy*dy/dx
    return np.hstack((dvdRb1*(150 - 50)/2., dvdRb2*(70 - 25)/2., dvdRf*(3 - .5)/2., dvdRc1*(2.5 - 1.2)/2.,\
        dvdRc2*(1.2 - .25)/2., dvdbeta*(300 - 50)/2.))
    
    