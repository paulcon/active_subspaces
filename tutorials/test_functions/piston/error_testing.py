# checking derivatives with first-order finite differences
import piston
import numpy as np      
h = 1e-6;
m = 7;
for k in range(0,20):
    x0 = 2*np.random.rand(1,m)-1
    x0 = x0.reshape(m)
    [f0,df] = piston.fun(x0)
    df_fd = np.zeros(m)
    for i in range(0,m):
        e = np.zeros(m)
        e[i] = 1
        [step,dif] = piston.fun(x0+h*e)
        df_fd[i] = (step-f0)/h
    print("PISTON: Norm of fd error:",np.linalg.norm(df-df_fd))