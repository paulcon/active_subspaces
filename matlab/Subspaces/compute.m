function sub = compute(X, f, df, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute the active and inactive subspaces from a collection of sampled
%   gradients.
%
%   Inputs:
%          X: M-by-m array that contains data points in the input space
%          f: M-by-1 array that contains evaluations of the function
%          df: M-by-m array of gradient evaluations
%          weights: (optional) M-by-1 array of weights
%                   Default: ones(M, 1)/M
%          sstype: (optional) integer indicating method for determine
%                  reduced dimension subspace
%                  Subspace types:
%                                 0, active subspace (Default)
%                                 1, normalized active subspace
%                                 2, active subspace x
%                                 3, normalized active subspace x
%                                 4, swarm subspace
%                                 5, ols, sdr
%                                 6, qphd, sdr
%                                 7, sir, sdr
%                                 8, phd, sdr
%                                 9, save, sdr
%                                 10, mave, sdr
%                                 11, opg, sdr
%          ptype: (optional) integer indicating method for determine
%                 dimension of active subspace
%                 Partition types:
%                                 0, eigenvalue gaps (Default)
%                                 1, response surface error bound
%                                 2, Li's ladle plot
%          n_boot: (optional) integer which gives the number of bootstrap
%                  replicates to use when computing bootstrap ranges.  If
%                  n_boot=0, then bootstrapping is not performed
%                  Default: 0
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

% Setup inputs
if ~isempty(X) && ~isempty(f)
    M = size(X, 1);
elseif ~isempty(df)
    M = size(df, 1);
else
    error('One of input/output pairs (X,f) or gradients (df) must not be empty')
end

if (length(varargin) >= 1) && ~isempty(varargin{1})
    weights = varargin{1};
    if any(size(weights) ~= [M, 1]) || any(weights < 0)
        error('Input weights must be M-by-1 array with non-negative entries')
    end
else
    weights = ones(M, 1)/M;
end

if (length(varargin) >= 2) && ~isempty(varargin{2})
    sstype = varargin{2};
    if (numel(sstype) > 1) || (rem(sstype, 1) > 0) || (sstype < 0)
        error('Input sstype must be non-negative integer')
    end
else
    sstype = 0;
end

if (length(varargin) >= 3) && ~isempty(varargin{3})
    ptype = varargin{3};
    if (numel(ptype) > 1) || (rem(ptype, 1) > 0) || (ptype < 0)
        error('Input ptype must be non-negative integer')
    end
else
    ptype = 0;
end

if (length(varargin) == 4) && ~isempty(varargin{4})
    n_boot = varargin{4};
    if (numel(n_boot) > 1) || (rem(n_boot, 1) > 0) || (n_boot < 0)
        error('Input n_boot must be non-negative integer')
    end
else
    n_boot = 0;
end

if length(varargin) > 4
    error('Too many inputs.')
end

% Preallocate space for 'sub' structure.
sub = struct('eigenvalues', [],...
             'eigenvectors', [],...
             'W1', [],...
             'W2', [],...
             'e_br', [],...
             'sub_br', []);

if (sstype == 0)
    if isempty(df)
        error('df is empty')
    end
    [e, W] = active_subspace(df, weights);
    ssmethod = @(X, f, df, weights) active_subspace(df, weights);
elseif (sstype == 1)
    if isempty(df)
        error('df is empty')
    end
    [e, W] = normalized_active_subspace(df, weights);
    ssmethod = @(X, f, df, weights) normalized_active_subspace(df, weights);
elseif (sstype == 2)
    if isempty(X) || isempty(df)
        error('X or df is empty')
    end
    [e, W] = active_subspace_x(X, df, weights);
    ssmethod = @(X, f, df, weights) active_subspace_x(X, df, weights);
elseif (sstype == 3)
    if isempty(X) || isempty(df)
        error('X or df is empty')
    end
    [e, W] = normalized_active_subspace_x(X, df, weights);
    ssmethod = @(X, f, df, weights) normalized_active_subspace_x(X, df, weights);
elseif (sstype == 4)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = swarm_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) swarm_subspace(X, f, weights);
elseif (sstype == 5)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = ols_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) ols_subspace(X, f, weights);
elseif (sstype == 6)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = qphd_subspace(X, f);
    ssmethod = @(X, f, df, weights) qphd_subspace(X, f);
elseif (sstype == 7)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = sir_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) sir_subspace(X, f, weights);
elseif (sstype == 8)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = phd_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) phd_subspace(X, f, weights);
elseif (sstype == 9)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = save_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) save_subspace(X, f, weights);
elseif (sstype == 10)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = mave_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) mave_subspace(X, f, weights);
elseif (sstype == 11)
    if isempty(X) || isempty(f)
        error('X or f is empty')
    end
    [e, W] = opg_subspace(X, f, weights);
    ssmethod = @(X, f, df, weights) opg_subspace(X, f, weights);
else
    error(['Unrecognized subspace type: ' num2str(sstype)])
end

sub.eigenvalues = e; sub.eigenvectors = W;

% Compute bootstrap ranges and partition
if (n_boot > 0)
    [e_br, sub_br, li_F] = bootstrap_ranges(e, W, X, f, df, weights, ssmethod, n_boot);
    sub.e_br = e_br; sub.sub_br = sub_br;
else
    if (ptype == 1) || (ptype == 2)
        error(['Need to run bootstrap for partition type ' num2str(ptype)])
    end
end

% Compute the partition
if (ptype == 0)
	n = eig_partition(e);
elseif (ptype == 1)
	n = errbnd_partition(e, sub_br(:, 2));
elseif (ptype == 2)
	n = ladle_partition(e, li_F);
else
	error(['Unrecognized partition type: ' num2str(ptype)])
end

sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:end);

end