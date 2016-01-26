function [e, W] = spectral_decomposition(df)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Use the SVD to compute the eigenvectors and eigenvalues for the active
%   subspace analysis
%
%   Inputs:
%          df: M-by-m array of gradient evaluations
%
%  Outputs:
%          e: m-by-1 array of eigenvalues
%          W: m-by-m array of eigenvectors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% M = number of samples; m = dimension of input space;
[M, m] = size(df);

% Compute the eigenvalues and eigenvectors which will ultimately form the
% active subspace.
[~, Sigma, W] = svd(df, 0);

if M >= m
    e = (diag(Sigma).^2)/M;
else
    e = [(diag(Sigma).^2)/M; zeros(m-M, 1)];
end
e(e < 0) = 0;

mult = sign(W(1, :));
mult(mult == 0) = 1;
W = W.*repmat(mult, m, 1);

end