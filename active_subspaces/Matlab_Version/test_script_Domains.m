clear variables
close all
clc

addpath 'Subspaces' 'Plotters' 'Domains'

%% Test function

% Data Set
[X, f, df] = test_function_2();
[M, m] = size(X);
n = 2;
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% Compute active subspace

sub = compute(df);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

Y = X*sub.W1;

%% Test functions to find zonotope vertices

[X_vert, Y_vert] = zonotope_vertices(sub.W1);

sufficient_summary(Y, f)
hold on
scatter(Y_vert(:, 1), Y_vert(:, 2), 'filled', 'k')

%% Test hit_and_run_z.m

for i = 1:M
    y = Y(i, :);
    Z = hit_and_run_z(20, y', sub.W1, sub.W2);
    
    x = repmat(y*sub.W1', 20, 1) + Z*sub.W2';
    if any(any(abs(x) > 1))
        'ERROR'
    end
    
    y_new = x*sub.W1;
    
    err(i) = sqrt(max(sum(abs(repmat(y, 20, 1) - y_new).^2, 2)));
end
['Max hit_and_run_z.m error: ' num2str(max(err))]

%% Test rejection_sampling_z.m

for i = 1:M
    y = Y(i, :);
    Z = rejection_sampling_z(20, y', sub.W1, sub.W2);
    
    x = repmat(y*sub.W1', 20, 1) + Z*sub.W2';
    if any(any(abs(x) > 1))
        'ERROR'
    end
    
    y_new = x*sub.W1;
    
    err(i) = sqrt(max(sum(abs(repmat(y, 20, 1) - y_new).^2, 2)));
end
['Max rejection_sampling_z.m error: ' num2str(max(err))]

%% Test random_walk_z.m

for i = 1:M
    y = Y(i, :);
    Z = random_walk_z(20, y', sub.W1, sub.W2);
    
    x = repmat(y*sub.W1', 20, 1) + Z*sub.W2';
    if any(any(abs(x) > 1))
        'ERROR'
    end
    
    y_new = x*sub.W1;
    
    err(i) = sqrt(max(sum(abs(repmat(y, 20, 1) - y_new).^2, 2)));
end
['Max random_walk_z.m error: ' num2str(max(err))]

%% Test rotate_x.m

Z = zeros(M, m-n, 10);
for i = 1:M
    Z0 = hit_and_run_z(10, Y(i, :)', sub.W1, sub.W2);
    
    Z(i, :, :) = reshape(Z0', 1, m-n, 10);
end

[X0, ind] = rotate_x(Y, Z, sub.eigenvectors);

if any(any(abs(X0) > 1))
    'ERROR'
end

err = sqrt(max(sum((Y(ind, :) - X0*sub.W1).^2, 2)));
['Max rotat_x.m error: ' num2str(err)]