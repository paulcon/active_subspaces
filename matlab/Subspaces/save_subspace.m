function [e, W] = save_subspace(X, f, weights)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Description
%
%   Inputs:
%          X: M-by-m array that contains data points in the input space
%          f: M-by-1 array that contains evaluations of the function
%          weights: M-by-1 array of weights
%
%  Outputs:
%          e: m-by-1 array of eigenvalues
%          W: m-by-m array of eigenvectors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M, m] = size(X);

if (max(max(X)) > 1) || (min(min(X)) < -1)
    gamma = 1;
else
    gamma = 1/3;
end

% Center and normalize data
Z = (1/sqrt(gamma))*(X - repmat(mean(X), M, 1));

% Bin data according to responses
H = 10;
bins = prctile(f, linspace(0, 100, H+1));
bins(1) = bins(1) - eps;

% Compute C matrix
C = zeros(m);
for i = 1:H
    in_slice = (f > bins(i)) & (f <= bins(i+1));
    n_h = sum(in_slice);
    if (n_h ~= 0)
        Z_tilde = Z(in_slice, :) - repmat(mean(Z(in_slice, :)), n_h, 1);
        sweights = weights(in_slice)/sum(weights(in_slice));
        if (n_h > 1)
            V = eye(m) - (Z_tilde'*(repmat(sweights, 1, m).*Z_tilde))/(1 - sum(sweights.^2));
        else
            V = eye(m);
        end
        C = C + V*V;
    end
end

[e, W] = sorted_eigh(C);

end