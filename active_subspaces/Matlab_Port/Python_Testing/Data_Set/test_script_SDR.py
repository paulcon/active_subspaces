import utils_data
import numpy as np
import active_subspaces.sdr
import matplotlib.pyplot as plt

# Set quantity of interest and input labels
QofI = 'QoI'
in_labels = ['rho', 'mu', 'dpdz', 'c_p', 'k', 'Pr_t']

# Set the number of parameter (m) and active subspace dimension (n)
m, n = 6, 3
# Load random sampling data
data = np.loadtxt('params_and_results.dat')
X_phys, f, df_phys = data[:, :m], data[:, m].reshape((data.shape[0], 1)), data[:, m+1:]
M = f.shape[0]

# Scale data so parameters are in [-1, 1]
X_norm, df_norm = utils_data.physical_to_normalized(X=X_phys, df=df_phys)

# Test linear gradient check
w = active_subspaces.sdr.linear_gradient_check(X_norm, f)

# Test quadratic model check
gamma = (1.0/3.0)*np.ones((1, m))
e, W = active_subspaces.sdr.quadratic_model_check(X_norm, f, gamma)

# Plot the active subspace info
active_subspaces.utils.plotters.eigenvalues(e, out_label=QofI)
active_subspaces.utils.plotters.eigenvectors(W[:, :n], in_labels=in_labels, out_label=QofI)
active_subspaces.utils.plotters.sufficient_summary(np.dot(X_norm, W[:, :2]), f, out_label=QofI)