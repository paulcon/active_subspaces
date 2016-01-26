function design = maximin_design(vert, N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Multivariate maximin design constrained by a polytope.
%
%   Inputs:
%          vert: nvp-by-m array containing vertices that define an
%                m-dimensional polytope
%          N: the number of points in the design
%
%  Outputs:
%          design: N-by-m array containing the design points in the
%                  polytope (does not contain the vertices)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(vert, 2);

% Construct polytope constraints for optimization.
K = convhulln(vert);
A = zeros(size(K, 1), n); b = zeros(size(K, 1), 1);
for i = 1:size(K, 1);
    c = null([vert(K(i, :), :), -ones(n, 1)]);
    
    if c(n+1) > 0
        A(i, :) = c(1:n)';
        b(i) = c(n+1);
    else
        A(i, :) = -c(1:n)';
        b(i) = -c(n+1);
    end
end
A1 = kron(eye(N), A);
b1 = kron(ones(N, 1), b);

fun = @(x) maximin_objective_function(x);

% Optimization options.
options = optimset('Algorithm', 'sqp', ...
                   'Display', 'Off', ...
                   'MaxIter', 100, ...
                   'TolFun', 1e-4);

warning('off','all')
design = []; fopt = 0; count = 0;
while (count < 3) || isempty(design)
    count = count + 1;
    
    % Get N initial points within the polytope.
    ind = [];
    while length(ind) < N
        x0 = randn(3*N, n);
        ind = find(all(A*x0'- repmat(b, 1, 3*N) < 0, 1), N, 'first');
        x0 = x0(ind,:);
    end
    
    % Find optimal design.
    [design_temp, fval] = fmincon(fun, x0, A1, b1, [], [], [], [], [], options);
    
    if (fval < fopt) && all(all(A*design_temp'- repmat(b, 1, N) < 0, 1))
        fopt = fval;
        design = design_temp;
    end
end
warning('on','all')

    % Object function for maximin optimization.  Returns the negative of
    % the minimum square distance between points in the design and the
    % polytope vertices.
    function f = maximin_objective_function(x)
        d0 = sum((kron(x, ones(N, 1)) - kron(ones(N, 1), x)).^2, 2);
        d0 = unique(d0);
        d0(d0 == 0) = [];
        [d0star, k0star] = min(d0);
        
        d1 = sum((kron(x, ones(size(vert, 1), 1)) ...
                        - kron(ones(N, 1), vert)).^2, 2);
        [d1star, k1star] = min(d1);
        
        g = zeros(N*n, 1);
        if d0star < d2star
            
        else
            
        end
        f = -min([d0; d1]);
        
        
%         f = -norm(x'*x, Inf);
    end
end