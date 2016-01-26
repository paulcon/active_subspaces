function [Yp, Yw] = zonotope_quadrature_rule(W1, N, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Quadrature when the dimension of the active subspace is greater than 1 
%   and the simulation parameter space is bounded.
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

addpath 'Domains' 'Designs'

[m, n] = size(W1);
if (n <= 1)
    error('ERROR: Dimension of active subspace must be greater than 1.')
end

% Build n-dimensional Delaunay triangulation.
[~, Y_vert] = zonotope_vertices(W1);
Y_pt = maximin_design(Y_vert, N);
Y = [Y_vert; Y_pt];
T = delaunayn(Y);

% Sample zonotope and find which simplexes contain which points.
Y_samples = (2*rand(NX, m) - 1)*W1;
T_ind = tsearchn(Y, T, Y_samples);

% Compute quadrature points and estimate the weights
Yp = zeros(size(T, 1), n); Yw = zeros(size(T, 1), 1);
for i = 1:size(T, 1)
    Yp(i, :) = mean(Y(T(i, :), :));
    Yw(i) = sum((T_ind == i))/NX;
end

end