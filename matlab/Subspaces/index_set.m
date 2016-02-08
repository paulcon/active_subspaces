function I = index_set(type,order,dim,constraint)
%INDEX_SET Builds the multi-indicies for orthogonal polynomials
%
% I = index_set(type,order)
% I = index_set(type,order,dimension)
% I = index_set(type,order,dimension,constraint)
%
% Constructs an array of nonnegative integers where each column of size
% 'dimension' contains a multi-index corresponding to a product type
% multivariate orthogonal polynomial. 
%
% Inputs
%   type:       A string dictating the type of basis set. The valid options
%               for type include: 'tensor' for a tensor product basis,
%               'full' or 'complete' or 'total order' for a full polynomial
%               basis, and 'constrained' for a custom constrained basis. 
%   
%   order:      For the cases of 'constrained' and 'full' (and equivalent)
%               types, the positive integer 'order' dictates the highest
%               order of polynomial in the basis. For type 'tensor', this
%               may be a vector of size 'dimension'.
%
% Optional inputs
%   dimension:  If type is 'tensor' and 'order' is a scalar, then the
%               dimension of the multi-indices must be specified.
%
%   constraint: A user-defined anonymous function that takes the form
%               @(index) constraint(index). It must take a valid
%               multi-index as its argument and return a number to be
%               compared to the given order. For example, the constraint
%               for the full polynomial basis is @(index) sum(index).
%
% Example:
%   % construct a total order basis
%   I = index_set('total order',4,2);
%   P = pmpack_problem('twobytwo',[0.2,0.6]);
%   X = spectral_galerkin(P.A,P.b,P.s,I);
%
% See also SPECTRAL_GALERKIN   


% Copyright 2009-2010 David F. Gleich (dfgleic@sandia.gov) and Paul G. 
% Constantine (pconsta@sandia.gov)
%
% History
% -------
% :2010-06-14: Initial release

if ~exist('dim','var') || isempty(dim) 
    dim=length(order);
else
    if length(order)>1 && dim~=length(order)
        error('dim is not consistent with order.');
    end
end

if isequal(type,'tensor')
    if isscalar(order), order=order*ones(dim,1); end
    I=1;
    for i=1:dim
        I=[kron(I,ones(order(i)+1,1)) kron(ones(size(I,1),1),(0:order(i))')];
    end
    I=I(:,2:end);
    I=I';
elseif isequal(type,'full') || isequal(type,'complete') || isequal(type,'total order')
    if ~isscalar(order), error('Order must be a scalar for constrained index set.'); end
    I=zeros(dim,1);
    for i=1:order
        II=full_index_set(i,dim);
        I=[I II']; %#ok<AGROW>
    end
elseif isequal(type,'constrained')
    if ~exist('constraint','var') || isempty(constraint), error('No constraint provided.'); end
    if ~isscalar(order), error('Order must be a scalar for constrained index set.'); end
    if order==0
        I=zeros(dim,1);
    else
        I=[];
        limit=order*ones(1,dim)+1;
        basis = cell(1,length(limit));
        for i=1:prod(limit)
            [basis{:}] = ind2sub(limit,i);
            b = cell2mat(basis)'-1;
            if constraint(b) <= order
                I=[I b];%#ok<AGROW>
            end
        end
    end
else 
    error('Unrecognized type: %s',type);
end

end

function I = full_index_set(n,d)

if d==1,
    I=n;
else
    I=[];
    for i=0:n
        II=full_index_set(n-i,d-1);
        m=size(II,1);
        T=[i*ones(m,1) II];
        I=[I; T]; %#ok<AGROW>
    end
end
end