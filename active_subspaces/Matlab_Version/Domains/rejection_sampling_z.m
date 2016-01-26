function [Z] = rejection_sampling_z(N, y, W1, W2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   A rejection sampling method for sampling the inactive variables from a
%   polytope.
%
%   Inputs:
%          N: integer giving the number of inactive variable samples
%          y: n-by-1 array giving a particle point in the active subspace
%          W1: m-by-n array containing the eigenvectors that define a basis
%              of the n-dimensional active subspace
%          W2: m-by-(m-n) array containing the eigenvectors that define a 
%              basis of the (m-n)-dimensional inactive subspace
%
%  Outputs:
%          Z: N-by-(m-n) array that contains values of the inactive
%          variable corresponding to the given value of the active variable
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(W1);

% Build a box around z for uniform sampling
s = W1*y;
A = [W2; -W2];
b = [-1-s; -1+s];

options = optimset('Display', 'Off');
lbox = zeros(1, m-n); ubox = zeros(1, m-n);
for i = 1:m-n
    clb = [zeros(i-1, 1); 1; zeros(m-n-i, 1)];
    lbox_temp = linprog(clb, -A, -b, [], [], [], [], [], options);
    lbox(i) = lbox_temp(i);
    
    cub = [zeros(i-1, 1); -1; zeros(m-n-i, 1)];
    ubox_temp = linprog(cub, -A, -b, [], [], [], [], [], options);
    ubox(i) = ubox_temp(i);
end

Zbox = repmat(ubox-lbox, 50*N, 1).*rand(50*N, m-n) + repmat(lbox, 50*N, 1);
ind = all(A*Zbox' >= repmat(b, 1, 50*N));

% Check that enough points were found.
if sum(ind) < N
    warning(['WARNING: rejection sampling only found ' num2str(sum(ind)) ' valid points.'])
end

% Grab first N (or all if less than N are found) valid points.
Z = Zbox(ind, :);
Z = Z(1:min([N, sum(ind)]), :);

end