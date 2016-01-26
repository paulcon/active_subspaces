clear variables
close all
clc

addpath 'Subspaces' 'Gradients'

%% Analytic Function Testing - EITHER THIS OR 'Data Set Testing' must be commented out

m = 5;
M = 100;
X_norm = 2*rand(M, m) - 1;

input_ranges = [0.5*ones(1, m); 3*ones(1, m)];

X_phys = zeros(size(X_norm));
for i = 1:M
    % Scale inputs so that they are on the interval [0.5, 3]
    X_phys(i, :) = (X_norm(i, :) + 1).*(input_ranges(2, :) - input_ranges(1, :))/2 + input_ranges(1, :);
end

[f, df] = analytic_function(X_phys);

for i = 1:M
    % Scale gradient to reflect inputs drawn from [-1, 1]^m domain.
    df(i, :) = df(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
    
    % Normalize gradient.  Not sure if this is really important
    df(i, :) = df(i, :)/norm(df(i, :));
end

%% Data Set Testing
% 
% data = importdata('test_data.dat');
% 
% % Set M=80 for full data set.
% M = 80;
% X_phys = data(1:M, 1:6);
% f = data(1:M, 7);
% df = data(1:M, 8:13);
% 
% input_ranges = [0.5, 5e-7, 2.5, 0.1, 5e-7, 0.1; ...
%                   2, 5e-6, 7.5,  10, 5e-6,  10];
% 
% [M, m] = size(df);
% 
% X_norm = zeros(size(X_phys));
% for i = 1:M
%     % Scale inputs so that they are on the interval [-1, 1]
%     X_norm(i, :) = 2*(X_phys(i, :) - input_ranges(1, :))./(input_ranges(2, :) - input_ranges(1, :)) - 1;
%     
%     % Apply scaling to gradient.
%     df(i, :) = df(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
%     
%     % Normalize gradient.  Not sure if this is really important
%     df(i, :) = df(i, :)/norm(df(i, :));
% end

%% True Gradients

% Compute and plot active subspace using gradients approximated by
% local_linear_gradients functions.
sub = compute(df);

% Plot results
opts = struct('title', 'True Gradients');
eigenvalues(sub.eigenvalues, sub.e_br, opts)
eigenvectors(sub.eigenvectors(:, 1:2), opts)
sufficient_summary(X_norm*sub.eigenvectors(:, 1:2), f, opts)

%% Local Linear Approximation Gradients

% Approximate gradients using local linear approximations.
df_local_linear = local_linear_gradients(X_phys, f);

for i = 1:size(df_local_linear,1)
    % Scale gradient to reflect inputs drawn from [-1, 1]^m domain.
    df_local_linear(i, :) = df_local_linear(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
    
    % Normalize gradient.  Not sure if this is really important.
     df_local_linear(i, :) = df_local_linear(i, :)/norm(df_local_linear(i, :));
end

% Compute and plot active subspace using gradients approximated by
% local_linear_gradients functions.
sub2 = compute(df_local_linear);

% Plot results
opts = struct('title', 'Local Linear Approx. Gradients');
eigenvalues(sub2.eigenvalues, sub2.e_br, opts)
eigenvectors(sub2.eigenvectors(:, 1:2), opts)
sufficient_summary(X_norm*sub2.eigenvectors(:, 1:2), f, opts)

%% Finite Difference Gradients

% Approximate gradients using finite differences.
fun = @(X) analytic_function(X);
df_fin_diff = finite_difference_gradients(X_phys, fun, 1e-10);

for i = 1:M
    % Scale gradient to reflect inputs drawn from [-1, 1]^m domain.
    df_fin_diff(i, :) = df_fin_diff(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
    
    % Normalize gradient.  Not sure if this is really important
    df_fin_diff(i, :) = df_fin_diff(i, :)/norm(df_fin_diff(i, :));
end

% Compute and plot active subspace using gradients approximated by
% local_linear_gradients functions.
sub3 = compute(df_fin_diff);

% Plot results
opts = struct('title', 'Finite Difference Gradients');
eigenvalues(sub3.eigenvalues, sub3.e_br, opts)
eigenvectors(sub3.eigenvectors(:, 1:2), opts)
sufficient_summary(X_norm*sub3.eigenvectors(:, 1:2), f, opts)