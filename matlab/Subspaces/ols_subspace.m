function [e, W] = ols_subspace(X, f, weights)

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

% Solve weighted least squares
A = [ones(M, 1), X].*repmat(sqrt(weights), 1, m+1);
b = f.*sqrt(weights);
u = A\b;
w = u(2:m+1);

C = w*w';

[e, W] = sorted_eigh(C);

end