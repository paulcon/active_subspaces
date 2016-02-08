function [Z] = hit_and_run_z(N, y, W1, W2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   A hit and run method for sampling the inactive variables from a
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

% Compute initial feasible point using the Chebyshev center.
s = W1*y;
normW2 = sqrt(sum(W2.^2, 2));
A = [W2, normW2; -W2, normW2];
b = [1-s; 1+s];
c = [zeros(m-n, 1); -1];

options = optimset('Display', 'Off');
zc = linprog(c, A, b, [], [], [], [], [], options);
z0 = zc(1:m-n);

% Define the polytope A >= b.
A = A(:, 1:m-n);
b = [-1-s; -1+s];

% Set tolerance.
ztol = 1e-6;
eps0 = ztol/4;

Z = zeros(N, m-n);
for i = 1:N
    
    % Search random directions.
    bad_dir = true;
    count = 0; maxcount = 50;
    while bad_dir
        d = randn(m-n, 1);
        bad_dir = any(A*(z0 + eps0*d) <= b);
        
        count = count + 1;
        if count >= maxcount
            warning(['There are no more directions worth pursuing in hit and run.  Got ' num2str(i) ' samples.'])
            Z(i:N, :) = repmat(z0', N-1, 1);
            return
        end
    end
    
    % Find constraints that impose lower and upper bounds on eps.
    f = b - A*z0;
    g = A*d;
    
    % Find an upper bound on the step
    min_ind = (g <= 0) & (f < -sqrt(eps));
    eps_max = min(f(min_ind)./g(min_ind));
    
    % Find a lower bound on the step
    max_ind = (g > 0) & (f < -sqrt(eps));
    eps_min = max(f(max_ind)./g(max_ind));
    
    % Randomly sample eps.
    eps1 = (eps_max - eps_min)*rand() + eps_min;
    
    % Take a step along d.
    z1 = z0 + eps1*d;
    Z(i, :) = z1';
end

end