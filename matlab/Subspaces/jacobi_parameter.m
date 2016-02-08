function s = jacobi_parameter(l,r,a,b)
% JACOBI_PARAMETER Construct a parameter with a Jacobi weight function
%
% s = jacobi_param() 
% s = jacobi_param(l,r,a,b) 
%
% Generates a parameter over the interval [l,r] with parameters for the
% Jacobi weight function alpha=a and beta=b. Default values are l=-1, r=1,
% a=0, and b=0, corresponding to a uniform weight of 0.5 over the interval
% [-1,1].
%
% See also PARAMETER, HERMITE_PARAMETER, LEGENDRE_PARAMETER

% Copyright 2009-2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
% Constantine (pconsta@sandia.gov)
%
% History
% -------
% :2010-06-14: Initial release


if ~exist('l','var') || isempty(l), l= -1; end
if ~exist('r','var') || isempty(r), r=  1; end
if ~exist('a','var') || isempty(a), a=  0; end
if ~exist('b','var') || isempty(b), b=  0; end

s.name = sprintf('jacobi(%g,%g) with support [%g,%g]', a, b, l, r);
s.recur = @(n) jacobi_recur(n,l,r,a,b);
s.l=l; s.r=r;
end

function ab=jacobi_recur(n,l,r,a,b)
% Compute the recurrence coefficients for the Jacobi polynomials
a0 = (b-a)/(a+b+2);
ab = zeros(n,2);
b2a2 = b^2 - a^2;
s = (r-l)/2; o = l + (r-l)/2;
if n>0
    ab(1,1) = s*a0+o;
    ab(1,2) = 1;
end
for k=2:n
    ab(k,1) = s*b2a2/((2*(k-1)+a+b)*(2*k + a+b))+o;
    if (k==2) 
        ab(k,2) = ((r-l)^2*(k-1)*(k-1+a)*(k-1+b)) / ...
                    ((2*(k-1)+a+b)^2*(2*(k-1)+a+b+1));
    else
        ab(k,2) = ((r-l)^2*(k-1)*(k-1+a)*(k-1+b)*(k-1+a+b)) / ...
                    ((2*(k-1)+a+b)^2*(2*(k-1)+a+b+1)*(2*(k-1)+a+b-1));
    end
end

end
