function [X, ind] = rotate_x(Y, Z, W)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Compute array of points in input space from points in active and
%   inactive subspaces.
%
%   Inputs:
%          Y: NY-by-n array containing points in the active subspace
%          Z: NY-by-(m-n)-by-NZ array containing points in the inactive
%             subspace corresponding to points in Y
%          W: m-by-m array containing the eigenvectors from the active
%             subspace analysis
%
%  Outputs:
%          X: (NY*NZ)-by-m array containing points in the input space
%          ind: (NY*NZ)-by-1 array containing indices for which rows of X
%               map to which rows of Y
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NY = size(Y, 1);
NZ = size(Z, 3);
m = size(W, 1);

YY = repmat(Y, [1, 1, NZ]);
YZ = reshape(permute(cat(2, YY, Z), [2, 3, 1]), m, NY*NZ)';
X = YZ*W';

ind = reshape(kron(1:NY, ones(NZ, 1)), NY*NZ, 1);

end