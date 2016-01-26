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

% Set value for number of bootstrap replicants.
if isempty(varargin)
    n_boot = 200;
elseif length(varargin) == 1
    n_boot = varargin{1};
    if ~isnumeric(n_boot) || rem(n_boot, 1) ~= 0 || (n_boot < 0)
        error('ERROR: n_boot must be a non-negative integer.')
    end
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

% Compute the eigenvalues and eigenvectors which will form basis for the 
% active and inactive subspaces.
[sub.eigenvalues, sub.eigenvectors] = spectral_decomposition(df);

% Compute bootstrap ranges for eigenvalues and subspace distances.
if n_boot > 0
    [sub.e_br, sub.sub_br] = bootstrap_ranges(df, sub.eigenvectors, n_boot);
end

% Determine dimension of the active subspace via a 'crappy heuristic'.
n = compute_partition(sub.eigenvalues);

sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:end);

end