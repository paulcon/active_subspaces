function [e, W] = quadratic_model_check(X, f, gamma)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Use the Hessian of a least-squares-fit quadratic model to identify
%   active and inactive subspaces.
%
%   Inputs:
%          X: M-by-m array containing sample points in the input space
%          f: M-by-1 array containing simulation evalutions corresponding
%             to the input points in X
%          gamma: m-by-1 array containing the variance of the input
%          parameters.  If inputs are drawn from uniform distribution on a
%          [-1, 1]^m hypercube, then gamma is a vector of 1/3's
%
%  Outputs:
%          e: m-by-1 array containing the eigenvalues of the quadratic
%             model's Hessian
%          W: m-by-m array containing the eigenvectors of the quadratic
%             model's Hessian
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath 'ResponseSurfaces'

[~, m] = size(X);

% Get regression coefficients.
[~, ~, g, H] = poly_train(X, f, 2);

% Compute eigenpairs.
[W, e] = eig(g*g' + H*diag(gamma)*H);

e = diag(e);
[e, ind] = sort(e, 'descend');
e(e < 0) = 0;

W = W(:, ind);
mult = sign(W(1, :));
mult(mult == 0) = 1;
W = W.*repmat(mult, m, 1);

end