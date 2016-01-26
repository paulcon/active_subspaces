function [X, Y] = zonotope_vertices(W1, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute the vertices of the zonotope.
%
%   Inputs:
%          W1: m-by-n array containing the eigenvectors that define a basis
%              of the n-dimensional active subspace
%          maxcount: (optional) max number of attempts at searching for
%                    zonotope vertices
%  Outputs:
%          Y: nzv-by-n matrix that contains the zonotope vertices in the
%             n-dimensional active subspaces
%          X: nzv-by-m matrix that contains the vertices of the
%             m-dimensional hypercube that map to the zonotope vertices.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    maxcount = 1e5;
elseif length(varargin) == 1
    maxcount = varargin{1};
    if ~isnumeric(maxcount) || rem(maxcount, 1) ~= 0 || (maxcount <= 0)
        error('ERROR: maxcount must be a positive integer.')
    end
else
    error('ERROR: Too many inputs.')
end

[m, n] = size(W1);

% Compute number of zonotope vertices
totalverts = nzv(m, n);

% Initialize search
Nsamples = 1e4;
Z = randn(Nsamples, n);

X = unique(sign(Z*W1'), 'rows');
X = unique([X; -X], 'rows');

N = size(X, 1);

count = 0;

% Search for additional zonotope vertices
while N < totalverts
    Z = randn(Nsamples, n);
    
    X0 = unique(sign(Z*W1'), 'rows');
    X0 = unique([X0; -X0], 'rows');
    
    X = unique([X; X0], 'rows');
    
    N = size(X, 1);
    
    count = count + 1;
    
    % Check if max number of iterations is exceeded.
    if count > maxcount
        break
    end
end

if size(X, 1) < totalverts
    warning(['WARNING: ' num2str(size(X, 1)) ' of ' num2str(totalverts) ' vertices found.'])
end

Y = X*W1;

end