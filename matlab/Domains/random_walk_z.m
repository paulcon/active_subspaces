function [Z] = random_walk_z(N, y, W1, W2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   A random walk method for sampling the inactive variables from a
%   polytope.
%
%   Inputs:
%          N: integer giving the number of inactive variable samples
%          y: n-by-1 array giving a particle point in the active subspace
%          W1: m-by-n array containing the eigenvectors that define a basis
%              of the n-dimensional active subspace
%          W2: m-by-(m-n) array containing the eigenvectors that define a 
%              basis of the (m-n)-dimensional inactive subspace
%
%  Outputs:
%          Z: N-by-(m-n) array that contains values of the inactive
%          variable corresponding to the given value of the active variable
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(W1);

s = W1*y;
 
% Get starting z0.
if all(zeros(m, 1) <= 1-s) && all(zeros(m, 1) >= -1-s)
    z0 = zeros(m-n, 1);
else
    lb = -ones(m, 1);
    ub = ones(m, 1);
    c = zeros(m, 1);
    
    options = optimset('Display', 'Off');
    x0 = linprog(c, [], [], W1', y, lb, ub, [], options);
    z0 = W2'*x0;
end

% Get MCMC step size.
sig = 0.1*min([norm(W2*z0+s-1), norm(W2*z0+s+1)]);

% Burn in
for i = 1:10*N
    zc = z0 + sig*randn(m-n, 1);
    if all(W2*zc <= 1-s) && all(W2*zc >= -1-s)
        z0 = zc;
    end
end

% Sample
Z = zeros(N, m-n);
for i = 1:N
    zc = z0 + sig*randn(m-n, 1);
    if all(W2*zc <= 1-s) && all(W2*zc >= -1-s)
        z0 = zc;
    end
    Z(i, :) = z0';
end

end