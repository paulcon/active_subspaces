import utils_data
import numpy as np
import active_subspaces
import matplotlib.pyplot as plt

# Set quantity of interest and input labels
QofI = 'QoI'
in_labels = ['rho', 'mu', 'dpdz', 'c_p', 'k', 'Pr_t']

# Set the number of parameter (m) and active subspace dimension (n)
m, n = 6, 2

# Load random sampling data
data = np.loadtxt('params_and_results.dat')
X_phys, f, df_phys = data[:, :m], data[:, m].reshape((data.shape[0], 1)), data[:, m+1:]
M = f.shape[0]

# Scale data so parameters are in [-1, 1]
X_norm, df_norm = utils_data.physical_to_normalized(X=X_phys, df=df_phys)

# Compute the active/inactive subspaces
sub = active_subspaces.subspaces.Subspaces()
sub.compute(df_norm/np.linalg.norm(df_norm, axis=1).reshape((df_norm.shape[0], 1)))

# Rewrite the active subspace to be 1-dimensional
sub.W1, sub.W2 = sub.eigenvectors[:, :n], sub.eigenvectors[:, n:]
sub.W1 = sub.W1.reshape(m, n)
sub.W2 = sub.W2.reshape(m, m-n)

dm = active_subspaces.domains.BoundedActiveVariableDomain(sub)
mp = active_subspaces.domains.BoundedActiveVariableMap(dm)

#Yp, Yw = active_subspaces.integrals.interval_quadrature_rule(mp, 5, NX=1000000)
Yp, Yw = active_subspaces.integrals.zonotope_quadrature_rule(mp, 10, NX=100000)
print Yp
print Yw