function [N] = nzv(m, n)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute the number of zonotope vertices for a linear map from R^m to
%   R^n.
%
%   Inputs:
%          m: integer giving the dimension of the hypercube
%          n: integer giving the dimenion of the low-dimensional subspace
%
%  Outputs:
%          N: integer giving the number of vertices defining the zonotope
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isnumeric(m) || (rem(m,1) ~= 0) || (m < 0)
    error('ERROR: m should be a positive integer')
end

if ~isnumeric(n) || (rem(n,1) ~= 0) || (n < 0)
    error('ERROR: n should be a positive integer')
end

% Compute number of zonotope vertices.
N = 0;
for i = 0:n-1
    N = N + nchoosek(m-1, i);
end
N = 2*N;

end