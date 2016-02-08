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

[N_vert, n] = size(vert);

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
                   'GradObj', 'On', ...
                   'MaxIter', 100, ...
                   'TolFun', 1e-4);

warning('off','all')
count = 0; x0 = []; design = []; fopt = 0;
while (count < 3) || isempty(design)
    count = count + 1;
    
    % Get N initial points within the polytope.
    ind = [];
    while (length(ind) < N)
        x0 = [x0; randn(3*N, n)];
        ind = find(all(A*x0'- repmat(b, 1, size(x0, 1)) < 0, 1), N, 'first');
        x0 = x0(ind,:);
        if (count >= 6)
            design = x0;
            break
        end
    end
    
    % Find optimal design.
    [design_temp, fval] = fmincon(fun, x0, A1, b1, [], [], [], [], [], options);
    
    tf_InPolytope = all(A*design_temp'- repmat(b, 1, N) < 0, 1);
    if all(tf_InPolytope)
        if (fval < fopt)
            fopt = fval;
            design = design_temp;
        end
        x0 = [];
    else
        x0 = design_temp(tf_InPolytope, :);
    end
end
warning('on','all')

    % Object function for maximin optimization.  Returns the negative of
    % the minimum square distance between points in the design and the
    % polytope vertices.
    function [f, g] = maximin_objective_function(x)
        % Compute distances between points.
        d0 = sum((kron(x, ones(N, 1)) - kron(ones(N, 1), x)).^2, 2);
        d0((1:N:N*N) + (0:N-1)) = 1e3;
        [d0star, k0star] = min(d0);
        % Compute distances between poitns and vertices.
        d1 = sum((kron(x, ones(N_vert, 1)) ...
                    - kron(ones(N, 1), vert)).^2, 2);
        [d1star, k1star] = min(d1);
        
        % Compute objective.
        f = -min([d0star; d1star]);
        
        % Compute gradient.
        g = zeros(N, n);
        if d0star < d1star
            istar = ceil(k0star/N);
            jstar = rem(k0star, N);
            if (jstar == 0)
                jstar = N;
            end
            
            g(istar, :) = 2*(x(istar, :) - x(jstar, :));
            g(jstar, :) = 2*(x(jstar, :) - x(istar, :));
        else
            istar = ceil(k1star/N_vert);
            jstar = rem(k1star, N_vert);
            if (jstar == 0)
                jstar = N_vert;
            end
            
            g(istar, :) = 2*(x(istar, :) - vert(jstar, :));
        end
        g = -g;
    end
end