import numpy as np

def parameter_bounds():
    # Order of params: mu, rho, dpdx, eta, B0
    return np.array([0.5, 5.0e-7, 2.5, 0.1, 5.0e-7, 0.1]), np.array([2.0, 5.0e-6, 7.5, 10.0, 5.0e-6, 10.0])

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