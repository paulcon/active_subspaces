import numpy as np

def func(data, QofI='U_avg'):
    M = data.shape[0]

    mu = data[:,0].reshape((M, 1))
    rho = data[:,1].reshape((M, 1))
    dpdx = data[:,2].reshape((M, 1))
    eta = data[:,3].reshape((M, 1))
    B0 = data[:,4].reshape((M, 1))
	
    Ha = B0/np.sqrt(eta*mu)
    mu0 = 1.0

    if (QofI == 'U_avg'):
        f = -dpdx*(eta*mu - Ha*eta*mu/np.tanh(Ha))/(mu*B0**2)
    elif (QofI == 'B_ind'):
        f = dpdx*mu0*(B0 - 2*np.sqrt(eta*mu)*np.tanh(Ha/2))/(2*B0**2)

    return f
    
def grad(data, QofI='U_avg'):
    M = data.shape[0]

    mu = data[:,0].reshape((M, 1))
    rho = data[:,1].reshape((M, 1))
    dpdx = data[:,2].reshape((M, 1))
    eta = data[:,3].reshape((M, 1))
    B0 = data[:,4].reshape((M, 1))
	
    Ha = B0/np.sqrt(eta*mu)
    mu0 = 1.0

    if (QofI == 'U_avg'):
        df_dmu = -dpdx*(np.sqrt(eta*mu)/np.tanh(Ha) - B0/np.sinh(Ha)**2)/(2*B0*mu**2)
        df_drho = np.random.uniform(1.0e-8, 1.0e-10, (M, 1))
        df_ddpdx = -(eta*mu - Ha*eta*mu/np.tanh(Ha))/(mu*B0**2)
        df_deta = -dpdx*(2*eta*mu - Ha*eta*mu/np.tanh(Ha) - (B0/np.sinh(Ha))**2)/(2*eta*mu*B0**2)
        df_dB0 = -dpdx*(-2*eta*mu + Ha*eta*mu/np.tanh(Ha) + (B0/np.sinh(Ha))**2)/(mu*B0**3)
    elif (QofI == 'B_ind'):
        df_dmu = -dpdx*mu0*(np.sqrt(eta*mu)*np.sinh(Ha) - B0)/(4*mu*(B0*np.cosh(Ha/2))**2)
        df_drho = np.random.uniform(1.0e-8, 1.0e-10, (M, 1))
        df_ddpdx = mu0*(B0 - 2*np.sqrt(eta*mu)*np.tanh(Ha/2))/(2*B0**2)
        df_deta = -dpdx*mu0*(np.sqrt(eta*mu)*np.sinh(Ha) - B0)/(4*eta*(B0*np.cosh(Ha/2))**2)
        df_dB0 = -dpdx*mu0*(B0 + B0/np.cosh(Ha/2)**2 - 4*np.sqrt(eta*mu)*np.tanh(Ha/2))/(2*B0**3)

    return np.concatenate((df_dmu, df_drho, df_ddpdx, df_deta, df_dB0), axis=1)

def parameter_bounds():
    # Order of params: mu, rho, dpdx, eta, B0
    return np.array([0.5, 0.5, 0.5, 0.5, 0.5]), np.array([3.0, 3.0, 3.0, 3.0, 3.0])

def physical_to_normalized(X=None, df=None):
    # Translate the parameters from their physical domain into a [-1,1]^m hypercube

    if (X is None) & (df is None):
        return None
    if X is not None:
        M = X.shape[0]
        X2 = np.zeros(X.shape)
    if df is not None:
        M = df.shape[0]
        df2 = np.zeros(df.shape)

    X_lower, X_upper = parameter_bounds()
	
    for r in range(M):
        if X is not None:
            X2[r,:] = 2/(X_upper - X_lower)*(X[r,:] - X_lower) - 1
        if df is not None:
            df2[r,:] = ((X_upper - X_lower)/2)*df[r,:]

    if (X is not None) & (df is not None):
        return X2, df2
    elif X is not None:
        return X2
    elif df is not None:
        return df2

def normalized_to_physical(X=None, df=None):
    # Translate the parameters from [-1,1]^m hypercube to the physical domain

    if (X is None) & (df is None):
        return None
    if X is not None:
        M = X.shape[0]
        X2 = np.zeros(X.shape)
    if df is not None:
        M = df.shape[0]
        df2 = np.zeros(df.shape)

    X_lower, X_upper = parameter_bounds()
	
    for r in range(M):
        if X is not None:
            X2[r,:] = ((X_upper - X_lower)/2)*(X[r,:] + 1) + X_lower
        if df is not None:
            df2[r,:] = (2/(X_upper - X_lower))*df[r,:]

    if (X is not None) & (df is not None):
        return X2, df2
    elif X is not None:
        return X2
    elif df is not None:
        return df2