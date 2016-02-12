function [e, W] = normalized_active_subspace(df, weights)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Description
%
%   Inputs:
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

% Compute norms of gradients
norm_df = sqrt(sum(df.^2, 2));

% Determine if any norms are close to zero
ind = (norm_df < sqrt(eps));

% Normalize gradients and set those with small norms to zero
df = df./repmat(norm_df, 1, m); df(ind, :) = 0;

% Compute the matrix
C = df'*(repmat(weights, 1, m).*df);

[e, W] = sorted_eigh(C);

end