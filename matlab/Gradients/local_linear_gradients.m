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
    error('Not enough samples for local linear models.')
end

if isempty(varargin)
    p = min([floor(1.7*m), M]);
    weights = ones(M, 1)/M;
elseif (length(varargin) == 1)
    if (numel(varargin{1}) == 1)
        p = varargin{1};
        if ~isnumeric(p) || (rem(p,1) ~= 0)
            error('Input p must be an integer.')
        elseif (p < m+1) || (p > M)
            error('Input p must be between m+1 and M.')
        end
        weights = ones(M, 1)/M;
    else
        weights = varargin{1};
        if any(size(weights) ~= [M, 1]) || any(weights < 0)
            error('Input weights must be M-by-1 array with non-negative entries.')
        end
        p = min([floor(1.7*m), M]);
    end
elseif (length(varargin) == 2)
    if (numel(varargin{1}) == 1)
        p = varargin{1};
        weights = varargin{2};
    else
        weights = varargin{1};
        p = varargin{2};
    end
    
    if ~isnumeric(p) || (rem(p,1) ~= 0)
        error('Input p must be an integer.')
    elseif (p < m+1) || (p > M)
        error('Input p must be between m+1 and M.')
    end
    
    if any(size(weights) ~= [M, 1]) || any(weights < 0)
        error('Input weights must be M-by-1 array with non-negative entries.')
    end
else
    error('Too many inputs.')
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
    D2 = sum((X - repmat(x, [M, 1])).^2, 2);
    [~, ind] = sort(D2);
    ind = ind(D2(ind) ~= 0);
    
    % Create linear regression using these p points.
    warning('off','all')
    A = [ones(p, 1), X(ind(1:p), :)].*repmat(sqrt(weights(ind(1:p))), 1, m+1);
    b = f(ind(1:p)).*sqrt(weights(ind(1:p)));
    u = A\b;
    warning('on','all')
    
    % Take gradient.
    df(i, :) = u(2:m+1)';
end

end