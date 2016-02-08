function s = parameter(type,l,r,a,b)
%PARAMETER Construct a parameter with a uniform weight function
%
% s = parameter()
% s = parameter(type,l,r,a,b);
%
% A wrapper function that constructs a parameter with a specified weight
% function as determined by the input string 'type'. The available options
% for 'type' are
%
%   Jacobi:     A parameter supported on the interval [l,r] with a general
%               Jacobi weight function.
%
%   Legendre:   A parameter supported on the interval [l,r] with a uniform
%               weight function.
%
%   Chebyshev:  A parameter supported on the interval [l,r] with a
%               Chebyshev weight function.
%
%   Hermite:    A parameter supported on the interval [-Inf,Inf] with a
%               standard Gaussian weight function.
%
%   Gaussian:   Equivalent to 'Hermite'.
%
% See also JACOBI_PARAMETER, HERMITE_PARAMETER, LEGENDRE_PARAMETER

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

if nargin==0, type='legendre'; end

switch lower(type)
    case 'jacobi'
        s = jacobi_parameter(l,r,a,b);
    case 'legendre'
        s = jacobi_parameter(l,r,0,0);
    case 'chebyshev'
        s = jacobi_parameter(l,r,-1/2,-1/2);
    case 'hermite'
        s = hermite_parameter();
    case 'gaussian'
        s = hermite_parameter();
    otherwise
        error('Unrecognized parameter type: %s',type);
end

end
