import numpy as np     
import active_subspaces as ac 
def piston(xx):
    #each row of xx should be [M, S, V0, k, P0, Ta, T0] in the normalized input space
    #returns a column vector of the piston function at each row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    M0 = x.shape[0]
    
    #Unnormalize inputs
    xl = np.array([30, .005, .002, 1000, 90000, 290, 340])
    xu = np.array([60, .02, .01, 5000, 110000, 296, 360])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    M = x[:,0]; S = x[:,1]; V0 = x[:,2]; k = x[:,3]
    P0 = x[:,4]; Ta = x[:,5]; T0 = x[:,6]
    
    A = P0*S + 19.62*M - k*V0/S
    V = S/(2*k)*(-A + np.sqrt(A**2 + 4*k*P0*V0*Ta/T0))
    pi = np.pi
    
    return (2*pi*np.sqrt(M/(k + S**2*P0*V0*Ta/(T0*V**2)))).reshape(M0, 1)
    
def piston_grad(xx):
    #each row of xx should be [M, S, V0, k, P0, Ta, T0] in the normalized input space
    #returns a matrix whose ith row is gradient of piston function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    
    #Unnormalize inputs
    xl = np.array([30, .005, .002, 1000, 90000, 290, 340])
    xu = np.array([60, .02, .01, 5000, 110000, 296, 360])
    x = ac.utils.misc.BoundedNormalizer(xl, xu).unnormalize(x)
    
    M = x[:,0]; S = x[:,1]; V0 = x[:,2]; k = x[:,3]
    P0 = x[:,4]; Ta = x[:,5]; T0 = x[:,6]
    
    A = P0*S + 19.62*M - k*V0/S
    V = S/(2*k)*(-A + np.sqrt(A**2 + 4*k*P0*V0*Ta/T0))
    pi = np.pi
    Q = k + S**2*P0*V0*Ta/(T0*V**2) #Convenience variables
    R = A**2 + 4*k*P0*V0*Ta/T0
        
    dCdM = (pi*(M*Q)**-.5 + 2*pi*M**.5*Q**-1.5*S**3*P0*V0*Ta/(2*k*T0*V**3)*(R**-.5*A*19.62 - 19.62))[:,None]
    dCdS = (-pi*M**.5*Q**-1.5*(2*S*P0*V0*Ta/(T0*V**2) - 2*S**2*P0*V0*Ta/(T0*V**3)*(V/S + S/(2*k)*(R**-.5*A*(P0 + k*V0/S**2) - P0 - k*V0/S**2))))[:,None]
    dCdV0 = (-pi*M**.5*Q**-1.5*(S**2*P0*Ta/(T0*V**2) - 2*S**3*P0*V0*Ta/(2*k*T0*V**3)*(R**-.5/2*(4*k*P0*Ta/T0 - 2*A*k/S) + k/S)))[:,None]
    dCdk = (-pi*M**.5*Q**-1.5*(1 - 2*S**2*P0*V0*Ta/(T0*V**3)*(-V/k + S/(2*k)*(R**-.5/2*(4*P0*V0*Ta/T0 - 2*A*V0/S) + V0/S))))[:,None]
    dCdP0 = (-pi*M**.5*Q**-1.5*(S**2*V0*Ta/(T0*V**2) - 2*S**3*P0*V0*Ta/(2*k*T0*V**3)*(R**-.5/2*(4*k*V0*Ta/T0 + 2*A*S) - S)))[:,None]
    dCdTa = (-pi*M**.5*Q**-1.5*(S**2*P0*V0/(T0*V**2) - 2*S**3*P0*V0*Ta/(2*k*T0*V**3)*(R**-.5/2*4*k*P0*V0/T0)))[:,None]
    dCdT0 = (pi*M**.5*Q**-1.5*(S**2*P0*V0*Ta/(T0**2*V**2) + 2*S**3*P0*V0*Ta/(2*k*T0*V**3)*(-R**-.5/2*4*k*P0*V0*Ta/T0**2)))[:,None]
    
    #The gradient components must be scaled in accordance with the chain rule: df/dx = df/dy*dy/dx
    return np.hstack((dCdM*(60 - 30)/2., dCdS*(.02 - .005)/2., dCdV0*(.01 - .002)/2., dCdk*(5000 - 1000)/2., dCdP0*(110000 - 90000)/2.,\
        dCdTa*(296 - 290)/2., dCdT0*(360 - 340)/2.))
    
    