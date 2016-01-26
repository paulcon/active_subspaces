function [f,df] = poly_predict(X,Coef,N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate the least-squares-fit polynomial approximation with monomial
%   basis containing all possible terms.
%
%   Inputs:
%          X: The M-by-m array of points to evaluate the polynomial
%             approximation such that m is the number of dimensions
%       Coef: The p-by-1 vector of estimated coefficients from the training
%             data for a particular Nth order polynomial
%
%   Outputs:
%          f: An M-by-1 array of predictions from the polynomial
%             approximation
%         df: An M-by-m array of gradient predictions from the polynomial
%             approximation 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determine number of observations and number of independent variables
[M,m] = size(X);

% Obtain combinations of powers
I = index_set('full',N,m);

% Obtain combinatinos of gradient powers
I_df = index_set('full',N-1,m);

% Determine resulting dimension of approximation
[~,p] = size(I);
[~,p_df] = size(I_df);

% Construct array of polynomial combinations for a monomial basis 
B   = zeros(M,p);
Bdf = zeros(M,p_df);
for i=1:p
    B(:,i)  = prod(bsxfun(@power,X,I(:,i)'),2);
    if i <= p_df
        Bdf(:,i)= prod(bsxfun(@power,X,I_df(:,i)'),2);
    end
end

IC = bsxfun(@times,I',Coef);
IC = IC';
C = zeros(m,length(find(IC))/m);
for i = 1:m
    j = find(IC(i,:));
    C(i,:) = IC(i,j);
end

% Calculate gradients
df = Bdf*C';

f = B*Coef;