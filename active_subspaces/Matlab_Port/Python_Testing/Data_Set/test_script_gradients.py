import utils_data
import numpy as np
import active_subspaces

# Set quantity of interest and input labels
QofI = 'QoI'
in_labels = ['rho', 'mu', 'dpdz', 'c_p', 'k', 'Pr_t']

# Set the number of parameter (m) and active subspace dimension (n)
m, n = 6, 1

# Load random sampling data
data = np.loadtxt('params_and_results.dat')
X_phys, f, df_phys = data[:, :m], data[:, m].reshape((data.shape[0], 1)), data[:, m+1:]
M = f.shape[0]

# Scale data so parameters are in [-1, 1]
X_norm, df_norm = utils_data.physical_to_normalized(X=X_phys, df=df_phys)

###############################################################################
##### Local Linear Approximation Gradients
###############################################################################

# Estimate gradients using local linear approximation
df_local_linear = active_subspaces.gradients.local_linear_gradients(X_phys, f)

# Scale data so parameters are in [-1, 1]
df_local_linear = utils_data.physical_to_normalized(df=df_local_linear)

# Compute the active/inactive subspaces
# NOTE: The gradient is normalized for this computation
sub = active_subspaces.subspaces.Subspaces()
sub.compute(df_local_linear/np.linalg.norm(df_local_linear, axis=1).reshape((df_local_linear.shape[0], 1)))

# Rewrite the active/inactive subspace variables to be n-dimensional
sub.W1, sub.W2 = sub.eigenvectors[:, :n], sub.eigenvectors[:, n:]
sub.W1 = sub.W1.reshape(m, n)
sub.W2 = sub.W2.reshape(m, m-n)

# Define the active/inactive variables
Y, Z = np.dot(X_norm, sub.W1), np.dot(X_norm, sub.W2)

# Plot the active subspace info
active_subspaces.utils.plotters.eigenvalues(sub.eigenvalues, e_br=sub.e_br, out_label='Local Linear Approximation')
active_subspaces.utils.plotters.eigenvectors(sub.W1, in_labels=in_labels, out_label='Local Linear Approximation')

###############################################################################
##### True Gradients
###############################################################################

# Compute the active/inactive subspaces
# NOTE: The gradient is normalized for this computation
sub = active_subspaces.subspaces.Subspaces()
sub.compute(df_norm/np.linalg.norm(df_norm, axis=1).reshape((M, 1)))

# Rewrite the active/inactive subspace variables to be n-dimensional
sub.W1, sub.W2 = sub.eigenvectors[:,:n], sub.eigenvectors[:,n:]
sub.W1 = sub.W1.reshape(m, n)
sub.W2 = sub.W2.reshape(m, m-n)

# Define the active/inactive variables
Y, Z = np.dot(X_norm, sub.W1), np.dot(X_norm, sub.W2)

# Plot the active subspace info
active_subspaces.utils.plotters.eigenvalues(sub.eigenvalues, e_br=sub.e_br, out_label='True')
active_subspaces.utils.plotters.eigenvectors(sub.W1, in_labels=in_labels, out_label='True')