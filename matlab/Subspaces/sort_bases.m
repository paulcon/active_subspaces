function Y=sort_bases(X)
%SORT_BASES Sorts the multivariate bases by total degree
%
% Y = sort_bases(X) 
%
% Sorts the basis elements of the expansion 'X' by total order and arrange
% the coefficients to match.

% Copyright 2009-2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
% Constantine (pconsta@sandia.gov)
%
% History
% -------
% :2010-06-14: Initial release


s=sum(X.index_set,1);
[t,ind]=sort(s,'ascend'); %#ok<ASGLU>
X.coefficients=X.coefficients(:,ind); 
X.index_set=X.index_set(:,ind);
Y=X;
