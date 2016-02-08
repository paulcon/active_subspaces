function [X,errz] = pseudospectral(iAb,s,pOrder,varargin)
%PSEUDOSPECTRAL Pseudospectral approximation of solution to A(s)x(s)=b(s)
%
% X = pseudospectral(iAb,s,pOrder);
% X = pseudospectral(iAb,s,pOrder,...);
% [X,err] = pseudospectral(iAb,s,pOrder,...);
%
% The function pseudospectral computes the Pseudospectral approximation to 
% the solution x(s) of the parameterized matrix equation A(s)x(s)=b(s) 
% using a basis of multivariate orthogonal polynomials. 
%
% Outputs:
%   X:          A struct containing the components of the Pseudospectral 
%               solution. See below for a more detailed description.
%
%   err:        An estimate of the error in the approximation. If pOrder 
%               is set to 'adapt', then this is a vector of the error 
%               estimates to examine convergence.
%
% Required inputs:
%   iAb:        A function handle of the form @(s) iAb(s) that returns the
%               solution of the parameterized matrix equation (or any
%               function, in general) given a point in the parameter space.
%
%   s:          A vector of parameter structs. The length of s is
%               considered to be the dimension d of the parameter space. 
%               See the function parameter.m.
%
%   pOrder:     The order of the polynomial approximation. A scalar input
%               creates a tensor product basis set for the given 
%               dimension of all the same order. A vector input creates a 
%               tensor product basis set with the order specified for each 
%               dimension by the components of pOrder. Set this to the 
%               string 'adapt' to increase the polynomial order of a tensor 
%               polynomial basis until the chosen error estimate is below a
%               given tolerance.
%
% Optional inputs:
% To specify optional inputs, use the 'key/value' format. For example, to
% set the convergence tolerance 'pTol' to 1e-6, include 'pTol',1e-6 in 
% the argument list. See the examples below for more details.
%
%   pTol:       A scalar representing the tolerance for the pOrder='adapt'
%               option. This is ignored if 'pOrder' is not set to 'adapt'.
%               (Default 1e-8)
%
%   ErrEst:     A string that determines the type of error estimate to use.
%               The options include: 'relerr' computes the difference 
%               between the approximation and a reference solution.
%               'mincoeff' computes the average of the magnitude of the
%               coefficients associated with the terms of the two
%               highest degrees. 'resid' uses the inputs 'A' and 'b' to
%               compute a residual error estimate. (Default 'relerr')
%
%   RefSoln:    A struct containing a reference solution to compare against
%               a computed approximation. If pOrder='adapt', then this is
%               set as the approximation with 'pOrder' one less than the
%               current approximation. (Default [])
%
%   MatFun:     A function handle that returns the matrix given a point in
%               the parameter space. Either this or 'MatVecFun' are 
%               required if the 'ErrEst' is set to 'Resid'. (Default [])
%
%   VecFun:     A function handle that returns the right hand side given a
%               point in the parameter space. This is required if the
%               'ErrEst' is set to 'Resid'. (Default [])
%
%   MatVecFun:  A function handle that returns the multiplication of the
%               matrix evaluated at a point in the parameter space
%               multiplied by a given vector. Either this or 'MatVec' are 
%               required if the 'ErrEst' is set to 'Resid'.(Default [])
%
%   Verbose:    A flag taking values 0 or 1 that tells the code whether or
%               not to print detailed status information during the
%               computation of the approximation. (Default 0)
%
% The output struct 'X' contains the following fields.
%   X.coefficients: An array of size N by # of bases containing the
%               coefficients of the Galerkin approximation.
%
%   X.index_set: An array of size d by # of bases containing the
%               multi-indicies corresponding to each basis polynomial.
%
%   X.variables: The input vector of parameters 's' used to construct the
%               approximation.
%
%   X.fun:      If 'X' is a pseudospectral approximation, this is the
%               anonymous function used to compute the coefficients.
%
%   X.matfun:   The function handle that returns the matrix at a given
%               point in the parameter space. 
%
%   X.vecfun:   The function handle that returns the right hand side at a 
%               given point in the parameter space. 
%
%   X.matvecfun: The function handle that returns the matrix at a given
%               point in the parameter space multiplied by a given vector.
%
% References:
%   Constantine, P.G., Gleich, D.F., Iaccarino, G. 'Spectral Methods for
%       Parameterized Matrix Equations'. SIMAX, 2010.
%       http://dx.doi.org/10.1137/090755965
%
% Example:
%   A = @(t) [2 t; t 1];                    % 2x2 parameterized matrix
%   b = @(t) [2; 1];                        % constant right hand side
%   iAb = @(t) A(t)\b(t);
%   s = parameter();                        % parameter defined on [-1,1]
%   pOrder = 13;                            % degree 13 approximation
%   X = pseudospectral(iAb,s,pOrder);
%   
% See also SPECTRAL_GALERKIN

%
% Copyright 2009-2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
% Constantine (pconsta@sandia.gov)
%
% History
% -------
% :2010-06-14: Initial release

if nargin<3, error('Not enough input arguments.'); end

dim=length(s); % dimension

% set default values
ptol=0;
errest='relerr'; % types: relerr, mincoeff, resid
refsoln=[];
matfun=[];
vecfun=[];
matvecfun=[];
verbose=0;
vprintf = @(varargin) fprintf('pseudospectral: %s\n',sprintf(varargin{:}));

errz=[];

for i=1:2:length(varargin)-1
    switch lower(varargin{i})
        case 'ptol'
            ptol=varargin{i+1};
        case 'errest'
            errest=lower(varargin{i+1});
        case 'refsoln'
            refsoln=varargin{i+1};
        case 'matfun'
            matfun=varargin{i+1};
        case 'vecfun'
            vecfun=varargin{i+1};
        case 'matvecfun'
            matvecfun=varargin{i+1};
        case 'verbose'
            verbose=varargin{i+1};
        otherwise
            error('Unrecognized option: %s\n',varargin{i});
    end
end

if ~verbose, vprintf = @(varargin) []; end

% Check to see whether or not we do a convergence study.
if isnumeric(pOrder)
    if isscalar(pOrder) 
        pOrder=pOrder*ones(dim,1); 
    else
        if max(size(pOrder))~=dim, error('Tensor order must equal dimension.'); end
    end
elseif isequal(pOrder,'adapt')
    if ptol==0, ptol=1e-8; end
else
    error('Unrecognized option for pOrder: %s\n',pOrder);
end

if isequal(pOrder,'adapt')
    vprintf('using adaptive computation ErrEst=%s',errest);
    
    if isequal(errest,'mincoeff') && ~isempty(refsoln)
        warning('pmpack:ignored','Reference solution will be ignored.');
    end
    
    if isempty(refsoln)
        vprintf('computing reference solution');
        
        refsoln=pseudospectral(iAb,s,0,...
            'matfun',matfun,'vecfun',vecfun,'matvecfun',matvecfun);
    end
    
    err=inf; order=1;
    while err>ptol
        vprintf('adaptive solution order=%i, error=%g\n',order, err);
        
        X=pseudospectral(iAb,s,order,varargin{:},...
            'matfun',matfun,'vecfun',vecfun,'matvecfun',matvecfun);
        err=error_estimate(errest,X,refsoln);
        errz(order)=err;
        order=order+1;
        if isequal(errest,'relerr'), refsoln=X; end
    end
else
    vprintf('constructing quadrature rule npoints=%i, max_porder=%i',...
        prod(pOrder+1), max(pOrder));
    
    % Construct the array of dim dimensional gauss points and the eigenvector
    % matrix of the multivariate Jacobi matrix.
    Q=cell(dim,1);
    q0=1;
    for i=1:dim
        Q{i}=jacobi_eigenvecs(s(i),pOrder(i)+1);
        q0=kron(q0,Q{i}(1,:));
    end 
    p=gaussian_quadrature(s,pOrder+1);
    
    % evaluate the first point, so we can get the size of the system
    u0 = q0(1)*iAb(p(1,:));
    N = size(u0,1);

    % Solve the parameterized matrix equation at each gauss point.
    gn=prod(pOrder+1);
    Uc=zeros(N,gn);
    Uc(:,1) = u0;
    vprintf('evaluating solution at %i points with parfor',gn);
    parfor i=2:gn
        Uc(:,i) = q0(i)*iAb(p(i,:)); %#ok<PFBNS>
    end
    vprintf('evaluation complete');
    
    % in theory, Matlab can do these inplace.
    Uc = Uc';
    Uc = kronmult(Q,Uc);
    Uc = Uc';
    

    % Construct the coefficients with the basis labels.
    X.coefficients=Uc; 
    X.index_set=index_set('tensor',pOrder);
    X.variables=s; 
    X.fun=iAb;
    X.matfun=matfun;
    X.vecfun=vecfun;
    X.matvecfun=matvecfun;
    X=sort_bases(X);
    
    if nargout==2
        errz=error_estimate('MinCoeff',X);
    end
    
end
vprintf('done');

end


