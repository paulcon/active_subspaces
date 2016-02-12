function [e, W] = normalized_active_subspace_x(X, df, weights)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Description
%
%   Inputs:
%          X: M-by-m array that contains data points in the input space
%          df: M-by-m array of gradient evaluations
%          weights: M-by-1 array of weights
%
%  Outputs:
%          e: m-by-1 array of eigenvalues
%          W: m-by-m array of eigenvectors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = size(df, 2);

% Compute norms of inputs and gradients
norm_X = sqrt(sum(X.^2, 2));
norm_df = sqrt(sum(df.^2, 2));

% Determine if any norms are close to zero
ind_X = (norm_X < sqrt(eps));
ind_df = (norm_df < sqrt(eps));

% Normalize gradients and set those with small norms to zero
X = X./repmat(norm_X, 1, m); X(ind_X, :) = 0;
df = df./repmat(norm_df, 1, m); df(ind_df, :) = 0;

% Compute the matrix
A = df'*(repmat(weights, 1, m).*X);
C = 0.5*(A + A');

[e, W] = sorted_eigh(C);

end