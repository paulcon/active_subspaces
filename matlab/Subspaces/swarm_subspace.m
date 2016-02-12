function [e, W] = swarm_subspace(X, f, weights)

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

% Integration weights
W = weights*weights';

% Compute distances between all points and remove small values
D2 = reshape(sum((kron(X, ones(M, 1)) - kron(ones(M, 1), X)).^2, 2), M, M);
ind = (D2 < sqrt(eps));
W(ind) = 0; D2(ind) = 1;

% all weights
A = W.*((repmat(f, 1, M) - repmat(f', M, 1)).^2)./D2;

C = zeros(m);
for i = 1:M
    P = X - repmat(X(i, :), M, 1);
    C = C + P'*(P.*repmat(A(:, i), 1, m));
end

[e, W] = sorted_eigh(C);

end