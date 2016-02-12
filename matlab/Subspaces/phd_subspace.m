function [e, W] = phd_subspace(X, f, weights)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Description
%
%   Inputs:
%          X: M-by-m array that contains data points in the input space
%          f: M-by-1 array that contains evaluations of the function
%          weights: M-by-1 array of weights
%
%  Outputs:
%          e: m-by-1 array of eigenvalues
%          W: m-by-m array of eigenvectors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M, m] = size(X);

if (max(max(X)) > 1) || (min(min(X)) < -1)
    gamma = 1;
else
    gamma = 1/3;
end

% Center data
Z = X - repmat(mean(X), M, 1);

% Compute C matrix
C = (1 / sqrt(gamma)) * (Z' * (repmat((f - mean(f)).*weights, 1, m) .* Z));

[e, W] = sorted_eigh(C);

end