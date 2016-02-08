function [p,w]=gaussian_quadrature(s,n,gridflag)
%GAUSSIAN_QUADRATURE Compute a Gaussian quadrature rule
% 
% [p,w] = gaussian_quadrature(parameter(),n);
% [p,w] = gaussian_quadrature([parameter(),parameter()],n);
% [p,w] = gaussian_quadrature(...,gridflag);
%
% Computes the points and weights of a tensor product Gaussian quadrature
% rule of order 'n' for the given parameters 's'.
%
% Outputs
%   p:      If 'gridflag' is 1, then 'p' is an array of the d-dimensional
%           points of the quadrature rule. If 'gridflag' is 0, this is a
%           cell array of the 1-dimensional points corresponding to each
%           element of 's' with the number of points dictated by each
%           corresponding element of 'n'. 
%
%   w:      The weights of the quadrature rule. If 'gridflag' is set to 1, 
%           then this is a single vector containing the weights associated
%           with each d-dimensional point. If 'gridflag' is 0, this is a
%           cell array with the weights for each 1-dimensional rule.
%
% Inputs:
%   s:      A vector of parameter structs. 
%
%   n:      If 'n' is a scalar, then it constructs the same number of
%           points for each element of 's'. If 'n' is a d-dimensional
%           vector of positive integers, then it constructs a separate
%           1-dimensional quadrature rule for each element of 'n' and each
%           element of 's'. 
%
%   gridflag: A flag taking values 0 or 1 that determines whether the code
%           outputs an array of d-dimensional points or a cell array of the
%           1-dimensional rules. (Default 1)
%   
% Example:
%   f = @(x) sin(pi*x(1))+cos(pi*x(2));     % integrate the given function
%   s = [parameter(); parameter()];
%   n = [3; 4];
%   [p,w] = gaussian_quadrature(s,n);
%   fint = 0;
%   for i=1:size(p,1)
%       fint = fint + f(p(i,:))*w(i);
%   end
%   fint
%   quad2d(@(X,Y) sin(pi.*X)+cos(pi.*Y),-1,1,-1,1) % compare with matlab quad
%
% See also PARAMETER JACOBI_PARAMETER HERMITE_PARAMETER 

% Copyright 2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
% Constantine (pconsta@sandia.gov).

if ~exist('gridflag','var') || isempty(gridflag), gridflag=1; end

if isstruct(s) 
    p = cell(size(s));
    w = cell(size(s));
    if isscalar(n), n=n*ones(size(s)); end
    for i=1:numel(s)
        [p{i},w{i}] = gaussian_quadrature(s(i).recur(n(i)),n(i));
    end

    if numel(p)==1,
        % dereference the cell if there is only one element
        p=p{1};
        w=w{1};
    else
        if gridflag
            % build a tensor grid of points
            pgrid=1; wgrid=1;
            for i=1:numel(p)
                pgrid=[kron(pgrid,ones(length(p{i}),1)) kron(ones(size(pgrid,1),1),p{i})];
                wgrid=kron(wgrid,w{i});
            end
            p=pgrid(:,2:end);
            w=wgrid;
        end
    end
else
    % s is a set of recursion coefficients
    if n>size(s,1)
        error('gaussrule:insufficientCoefficients',...
            ['The gaussrule needs more coefficients than points,' ...
                'but numcoeff=%i and numpoints (n)=%i'], size(s,1), n);
    end
        
    ab=s;
    J=jacobi_matrix(s,n); 
    [V,D]=eig(J);
    [p,I]=sort(diag(D));
    w=V(1,I)'.^2;
    w=ab(1,2)*w;
end
        
