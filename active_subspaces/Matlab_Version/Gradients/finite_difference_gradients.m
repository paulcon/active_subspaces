function [df] = finite_difference_gradients(X, fun, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Estimate a collection of gradients from input/output pairs.
%
%   Inputs:
%          X: M-by-m array containing the the points at which to estimate
%             the gradients with finite differences
%          fun: function handle which accepts input values and returns
%               scalar simulation evaluations
%          h: (optional) scalar which gives the finite difference step size
%             Default: 1e-6
%
%  Outputs:
%          df: M-by-m matrix containing estimated gradients using finite
%          differences
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    h = 1e-6;
elseif length(varargin) == 1
    h = varargin{1};
    if (numel(h) ~= 1) || ~isnumeric(h) || (h <= 0)
        error('ERROR: h must be positive scalar')
    end
else
    error('ERROR: Too many inputs')
end

[M, m] = size(X);

% Build (M*(m+1))-by-m array of all inputs points needed to perform finite
% differences.
XX = kron(ones(m+1, 1), X) + h*kron([zeros(1, m); eye(m)], ones(M, 1));

% Evaluate fucntion at input values.
f = fun(XX);

% Compute finite difference gradients.
df = (reshape(f(M+1:end), M, m) - repmat(f(1:M), 1, m))/h;

end