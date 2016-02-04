function J=jacobi_matrix(s,n)
%JACOBI_MATRIX Construct a Jacobi matrix for a parameter
%
% J = jacobi_matrix(s,n)
%
% Given a single parameter 's' and a scalar 'n', this function constructs
% the symmetric, tridiagonal matrix of recursion coefficients for the
% orthogonal polynomial basis associated with the measure on the space of
% 's'. 
%
% Example:
%   s = legendre_parameter();
%   J = jacobi_matrix(s,5);
%
% See also JACOBI_EIGENVECS JACOBI_MATRIX

% Copyright 2009-2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
% Constantine (pconsta@sandia.gov)
%
% History
% -------
% :2010-06-14: Initial release

if ~isstruct(s) && (~exist('n','var') || isempty(n)), n=size(s,1); end
if isstruct(s)
    ab = s.recur(n);
elseif size(s,2)==2 && isfloat(s)
    ab = s;
    if n>size(ab,1)
        error('jacobi_matrix:insufficientCoefficients', ...
            ['Please increase the number of recursion coefficients' ...
                ' (currently %i) to construct the matrix of size %i'], ...
                size(ab,1), n);
    end
else 
    error('jacobi_matrix:invalidArgument', ...
        ['The parameter or set of recursion coefficients must be a struct',...
            ' or a n-by-2 matrix.']);
end
J = zeros(n,n);
% special case for n=1
if n==1, J = ab(1,1); return; end

J(1,1)=ab(1,1);
J(1,2)=sqrt(ab(2,2));
for i=2:n-1
    J(i,i)=ab(i,1);
    J(i,i-1)=sqrt(ab(i,2));
    J(i,i+1)=sqrt(ab(i+1,2));
end
J(n,n)=ab(n,1);
J(n,n-1)=sqrt(ab(n,2));
