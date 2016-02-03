import numpy as np

def fun(xx):

    a = xx[0];
    b = xx[1];
    c = xx[2];
    
    f = a**3+a*b-a*b*c
    
    df = np.array([(3*a**2 -b*c + b), a*(1-c), -a*b ])
    
    return [f,df]