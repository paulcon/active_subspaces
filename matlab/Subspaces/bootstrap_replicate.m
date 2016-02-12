function [X0, f0, df0, weights0] = bootstrap_replicate(X, f, df, weights)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Description
%
%   Inputs:
%          X: M-by-m array that contains data points in the input space
%          f: M-by-1 array that contains evaluations of the function
%          df: M-by-m array of gradient evaluations
%          weights: M-by-1 array of weights
%
%  Outputs:
%          n: integer which gives the dimension of the active subspace
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = size(weights, 1);
ind = randi(M, M, 1);

X0 = []; f0 = []; df0 = [];

if ~isempty(X) 
    X0 = X(ind, :);
end

if ~isempty(f)
    f0 = f(ind);
end

if ~isempty(df)
    df0 = df(ind, :);
end

weights0 = weights(ind);

end