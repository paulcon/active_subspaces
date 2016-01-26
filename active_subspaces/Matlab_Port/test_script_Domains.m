clear variables
close all
clc

addpath 'Subspaces' 'Plotters' 'Domains'

%% Analytic Function Testing - EITHER THIS OR 'Data Set Testing' must be commented out
% 
% m = 5; n = 2;
% M = 100;
% X_norm = 2*rand(M, m) - 1;
% 
% input_ranges = [0.5*ones(1, m); 3*ones(1, m)];
% 
% X_phys = zeros(size(X_norm));
% for i = 1:M
%     % Scale inputs so that they are on the interval [0.5, 3]
%     X_phys(i, :) = (X_norm(i, :) + 1).*(input_ranges(2, :) - input_ranges(1, :))/2 + input_ranges(1, :);
% end
% 
% [f, df] = analytic_function(X_phys);
% 
% for i = 1:M
%     % Scale gradient to reflect inputs drawn from [-1, 1]^m domain.
%     df(i, :) = df(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
%     
%     % Normalize gradient.  Not sure if this is really important
%     df(i, :) = df(i, :)/norm(df(i, :));
% end

%% Data Set Testing

data = importdata('test_data.dat');

% Set m=80 for full data set.
m = 80;
% Set dimension of active subspace.
n = 2;

X_phys = data(1:m, 1:6);
f = data(1:m, 7);
df = data(1:m, 8:13);

input_ranges = [0.5, 5e-7, 2.5, 0.1, 5e-7, 0.1; ...
                  2, 5e-6, 7.5,  10, 5e-6,  10];

[M, m] = size(df);
X_norm = zeros(size(X_phys));
for i = 1:M
    % Scale inputs so that they are on the interval [-1, 1]
    X_norm(i, :) = 2*(X_phys(i, :) - input_ranges(1, :))./(input_ranges(2, :) - input_ranges(1, :)) - 1;
    
    % Apply scaling to gradient.
    df(i, :) = df(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
    
    % Normalize gradient.  Not sure if this is really important
    df(i, :) = df(i, :)/norm(df(i, :));
end

%% Compute active subspace

% Compute and plot active subspace using gradients approximated by
% local_linear_gradients functions.
sub = compute(df);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

Y = X_norm*sub.W1;

%% Test functions to find zonotope vertices

[X_vert, Y_vert] = zonotope_vertices(sub.W1);

sufficient_summary(X_norm*sub.W1, f)
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