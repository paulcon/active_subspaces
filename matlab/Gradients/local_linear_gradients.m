function [df] = local_linear_gradients(X, f, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Estimate a collection of gradients from input/output pairs.
%
%   Inputs:
%          X: M-by-m array containing the m-dimensional inputs
%          f: M-by-1 array containing scalar outputs
%          p: (optional) integer which gives the number of nearest
%             neighbors to use when constructing the local linear model
%             Default: floor(1.7*m)
%
%  Outputs:
%          df: MM-by-m matrix containing estimated gradients using the local
%              linear model. The number of rows depends on the dimension of
%              the input space and the number of samples.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M, m] = size(X);

if M <= m
    error('ERROR: Not enough samples for local linear models.')
end

if isempty(varargin)
    p = min([floor(1.7*m), M]);
elseif length(varargin) == 1
    p = varargin{1};
    if ~isnumeric(p) || rem(p,1) ~= 0
        error('ERROR: p must be an integer.')
    elseif (p < m+1) || (p > M)
        error('ERROR: p must be between m+1 and M.')
    end
else
    error('ERROR: Too many inputs.')
end

% Determine number of gradients that can be returned giving M samples from
% the m-dimensional input space.
MM = min([ceil(10*m*log(m)), M-1]);

df = zeros(MM, m);
for i = 1:MM
    % Select one of the given samples points at random.
    ii = randi(MM, 1);
    x = X(ii,:);
    
    % Find p closest points to x.
    [~, ind] = sort(sum((X - repmat(x, [M, 1])).^2, 2));
    
    % Create linear regression using these p points.
    A = [ones(p, 1), X(ind(2:p+1), :)];
    b = f(ind(2:p+1));
    u = A\b;
    
    % Take gradient.
    df(i, :) = u(2:m+1)';
end

end