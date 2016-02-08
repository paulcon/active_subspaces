function [Yp, Yw] = interval_quadrature_rule(W1, N, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Quadrature when the dimension of the active subspace is 1 and the
%   simulation parameter space is bounded.
%
%   Inputs:
%          W1: m-by-n array containing the eigenvectors that define a basis
%              of the n-dimensional active subspace
%          N: the number of quadrature nodes in the active variables
%          NX: (optional) the number of samples to use to estimate the
%              quadrature weights
%              Default: 100,000
%
%  Outputs:
%          Yp: quadrature nodes on the active variables
%          Yw: quadrature weights on the active variables
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    NX = 1e5;
elseif length(varargin) == 1
    NX = varargin{1};
    if ~isnumeric(NX) || rem(NX, 1) ~= 0 || (NX <= 0)
        error('ERROR: NX must be a positive integer.')
    end
else
    error('ERROR: Too many inputs.')
end

addpath 'Domains'

[m, n] = size(W1);
if (n ~= 1)
    error('ERROR: Dimension of active subspace must be 1.')
end

% Compute quadrature points.
[~, Y] = zonotope_vertices(W1);
Y = sort(Y);
Y = linspace(Y(1)*(1+eps), Y(2)*(1+eps), N+1)';
Yp = 0.5*(Y(2:N+1) + Y(1:N));

% Estimate quadrature weights.
Y_samples = (2*rand(NX, m) - 1)*W1;
Yw = zeros(N, 1);
for i = 1:N
    Yw(i) = sum((Y_samples >= Y(i)) & (Y_samples < Y(i+1)));
end
Yw = Yw/NX;

end