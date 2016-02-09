function sub = compute(df, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute the active and inactie subspaces from a collection of sampled
%   gradients.
%
%   Inputs:
%          df: M-by-m array of gradient evaluations
%          n_boot: (optional) integer which gives the number of bootstrap
%                  replicates to use when computing bootstrap ranges.  If
%                  n_boot=0, then bootstrapping is not performed
%                  Default: 200
%          F: (optional) an ndarray of size M that contains evaluations of the function.
%          X: (optional) an ndarray of size M-by-m that contains data points in the input space.
%          fun: (optional) a specified function that outputs f(x), and df(x) the gradient vector for a data point x
%          c_index: (optional) an integer specifying which C matrix to compute, the default matrix is 0.
%          comp_flag: (optional) an integer specifying computation method: 0 for monte carlo, 1 for LG quadrature.
%          N: (optional) number of quadrature points per dimension.
%                  
%
%  Outputs:
%          sub: structure containing the following fields
%              eigenvalues: m-by-1 array of eigenvalues
%              eigenvectors: m-by-m array of eigenvectors
%              W1: m-by-n array containing the basis for the active
%                  subspace
%              W2: m-by-(m-n) array containing the basis for the inactive
%                  subspace
%              e_br: m-by-2 array containing the bootstrap ranges for the
%                    eigenvalues
%              sub_br: m-by-3 array containing the bootstrap ranges (first
%                      and third columns) and the mean (second column) of
%                      the error in the estimated subpsaces approximated by
%                      bootstrapping
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set varable arguements
if isempty(varargin)
    n_boot = 200;
    F = 0;
    X = 0;
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 1
    n_boot = varargin{1};
    F = 0;
    X = 0;
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
    if ~isnumeric(n_boot) || rem(n_boot, 1) ~= 0 || (n_boot < 0)
        error('ERROR: n_boot must be a non-negative integer.')
    end
elseif length(varargin) == 2
    n_boot = varargin{1};
    F = varargin{2};
    X = 0;
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 3
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 4
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    fun = varargin{4};
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 5
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    fun = varargin{4};
    c_index = varargin{5};
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 6
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    fun = varargin{4};
    c_index = varargin{5};
    comp_flag = varargin{6};
    N = 5;
elseif length(varargin) == 7
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    fun = varargin{4};
    c_index = varargin{5};
    comp_flag = varargin{6};
    N = varargin{7};
else
    error('ERROR: Too many inputs.')
end
% Preallocate space for 'sub' structure.
sub = struct('eigenvalues', [],...
             'eigenvectors', [],...
             'W1', [],...
             'W2', [],...
             'e_br', [],...
             'sub_br', []);
 %compute eigenvalues and eigenvecs
% n_boot = n_boot
% sizeF = size(F)
% sizeX = size(X)
% func = fun
% c_index = c_index
% comp_flag = comp_flag
% N = N
        if c_index == 0 && comp_flag == 0
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);
            % compute bootstrap ranges for eigenvalues and subspace distances
            if n_boot > 0
                [sub.e_br, sub.sub_br] = bootstrap_ranges(df,sub.eigenvectors,n_boot,F,X,c_index);
            end
        elseif c_index == 0 && comp_flag == 1
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);       
        end
        if c_index == 1 && comp_flag == 0
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);
            % compute bootstrap ranges for eigenvalues and subspace distances
            if n_boot > 0
                [sub.e_br, sub.sub_br] = bootstrap_ranges(df,sub.eigenvectors,n_boot,F,X,c_index);
            end
        elseif c_index == 1 && comp_flag == 1
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);       
        end
        if c_index == 2 && comp_flag == 0
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);
            % compute bootstrap ranges for eigenvalues and subspace distances
            if n_boot > 0
                [sub.e_br, sub.sub_br] = bootstrap_ranges(df,sub.eigenvectors,n_boot,F,X,c_index);
            end
        elseif c_index == 2 && comp_flag == 1
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);       
        end
        if c_index == 3 && comp_flag == 0
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);
            % compute bootstrap ranges for eigenvalues and subspace distances
            if n_boot > 0
                [sub.e_br, sub.sub_br] = bootstrap_ranges(df,sub.eigenvectors,n_boot,F,X,c_index);
            end
        elseif c_index == 3 && comp_flag == 1
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);       
        end
        if c_index == 4 && comp_flag == 0
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);
            % compute bootstrap ranges for eigenvalues and subspace distances
            if n_boot > 0
                [sub.e_br, sub.sub_br] = bootstrap_ranges(df,sub.eigenvectors,n_boot,F,X,c_index);
            end
        elseif c_index == 4 && comp_flag == 1
            [sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df,F,X,fun,c_index,comp_flag,N);       
        end





%%%%%%

% Compute the eigenvalues and eigenvectors which will form basis for the 
% active and inactive subspaces.
%[sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df);

% Compute bootstrap ranges for eigenvalues and subspace distances.
%if n_boot > 0
%    [sub.e_br, sub.sub_br] = bootstrap_ranges(df, sub.eigenvectors, n_boot);
%end

% Determine dimension of the active subspace via a 'crappy heuristic'.
n = compute_partition(sub.eigenvalues);

sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:end);

end