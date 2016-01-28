function [e,W,P,Et_df,E] = data_projections(X,fun,k,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Estimate eigenvectors of the data projections for compressed gradient
%   sampling
%
%    Inputs:
%           X: The M-by-m array of training points for the polynomial
%              approximation such that m is the number of dimensions
%           f: The M-by-1 vector of function values paired with the M
%              observations of the training points in X
%           k: Scaler value indicating the number of finite difference 
%              approximations in a single compressed gradient measurement
%           h: (Optional) differencing scale for a 1st order gradient 
%              approximation
%
%   Outputs:
%           e: m-by-1 array of eigenvalues
%           W: m-by-m array of eigenvectors
%           P: Projection array of compressed gradient
%       Et_df: Compressed gradient samples
%           E: Compressed gradient perturbations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    h = 0.001;
elseif length(varargin) == 1
    h = varargin{1};
end

% Determine number of measurements and absolute dimensions
[M,m] = size(X);

% Constrain user against rank deficient E'*E
if k > m
    disp('Error: k must be less than or equal to m');
    e = []; W = []; P = []; Et_df = []; E = []; 
else
    % Construct the compressed gradient perturbations
    E = randn(m,k,M);

    % Construct the the compressed sample
    Et_df = zeros(k,1);
    P     = zeros(M,m);
    for i = 1:M
        for j=1:k
            % Sample the user provided function
            Et_df(j,i) = 1/h*(fun(X(i,:)' + h*E(:,j,i))-fun(X(i,:)'));
        end

        % Projection of compressed gradient measurement
        P(i,:) = E(:,:,i)*((E(:,:,1)'*E(:,:,1))\Et_df(:,1));
    end

    % Run subspaces spectral decomposition
    [e, W] = spectral_decomposition(P);
end