function [Coef, B, g, H, f_hat, r] = poly_train(X, f, N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Train the least-squares-fit polynomial approximation with monomial
%   basis containing all possible terms.
%
%   Inputs:
%          X: The M-by-m array of training points for the polynomial
%             approximation such that m is the number of dimensions
%          f: The M-by-1 vector of function values paired with the M
%             observations of the training points in X
%          N: An integer indicating the highest order of the polynomial
%
%   Outputs:
%       Coef: The p-by-1 vector of coefficients obtained using the least
%             squares fit approximation
%          B: The M-by-p array of monomial basis combinations of the
%             observations in X
%          g: The m-by-1 vector of coefficients corresponding to the
%             degree 1 monomials in the polynomial approximation
%          H: The m-by-m array of coefficients corresponding to the degree
%             2 monomials in the approximation
%      f_hat: The M-by-1 approximation of the function evaluations at the
%             training points
%          r: The M-by-1 residuals defining the difference between the
%             approximations (f_hat) and the true function evaluations (f)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determine number of observations and number of independent variables
[M, m] = size(X);

% Obtain combinations of powers
I = index_set('full', N, m);

% Determine resulting dimension of approximation
[~, p] = size(I);

% Construct array of polynomial combinations for a monomial basis 
B = zeros(M, p);
for i = 1:p
    B(:, i) = prod(bsxfun(@power, X, I(:, i)'), 2);
end

% Calculate coefficients 
Coef = (B'*B)\(B'*f);

% Calculate 
f_hat = B*Coef;

r = f_hat-f;

% Organize Quadratic Forms
if N==2
    % Get linear coefficients
    [r, c] = find(I(:, 2:m+1));
    g = Coef(2:m+1);
    g = g(r(c));

    % Get quadratic coefficients
    quad_ind = (1:m*(m+1)/2) + (m + 1);
    [r, c] = find(I(:, quad_ind));
    H = zeros(m, m);
    for i = 1:m*(m+1)/2
        tf = (c == i);
        index = r(tf);
        if sum(tf) == 1
            H(index, index) = 2*Coef(m+1+i);
        else
            H(index(1), index(2)) = Coef(m+1+i);
            H(index(2), index(1)) = Coef(m+1+i);
        end
    end
else
    g = [];
    H = [];
end

end