function [n] = compute_partition(eigenvalues)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   A heuristic based on eigenvalue gaps for deciding the dimension of the
%   active subspace.
%
%   Inputs:
%          eigenvalues: m-by-1 array of eigenvalues
%
%  Outputs:
%          n: integer which gives the dimension of the active subspace
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Handle zero eigenvalues for the logarithm.
eigenvalues(eigenvalues == 0) = 1e-100;

% 'Crappy threshold for choosing active subspace dimension'
%   - Dr. Paul Constantine
[~, n] = max(abs(diff(log(eigenvalues))));

end