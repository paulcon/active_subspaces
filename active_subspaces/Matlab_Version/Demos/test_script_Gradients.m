clear variables
close all
clc

addpath 'Subspaces' 'Gradients' 'Plotters'

%% Test functions - Choose 1

% Linear function
m = 5; n = 1;
M = 100;
X = 2*rand(M, m) - 1;
[f, df] = test_function_1(X);
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

% Data Set - Cannot perform finite differences with a data set
% [X, f, df] = test_function_2();
% [M, m] = size(X);
% n = 2;
% df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% True Gradients

sub = compute(df);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

% Plot results
opts = struct('title', 'True Gradients');
eigenvalues(sub.eigenvalues, sub.e_br, opts)
eigenvectors(sub.W1, opts)

%% Local Linear Approximation Gradients

% Approximate gradients using local linear approximations.
df_local_linear = local_linear_gradients(X, f);
df_local_linear = df_local_linear./repmat(sqrt(sum(df_local_linear.^2, 2)), 1, m);

sub2 = compute(df_local_linear);
sub2.W1 = sub2.eigenvectors(:, 1:n);
sub2.W2 = sub2.eigenvectors(:, n+1:m);

% Plot results
opts = struct('title', 'Local Linear Approx. Gradients');
eigenvalues(sub2.eigenvalues, sub2.e_br, opts)
eigenvectors(sub2.W1, opts)

%% Finite Difference Gradients

% Approximate gradients using finite differences.
fun = @(X) test_function_1(X);
df_fin_diff = finite_difference_gradients(X, fun);
df_fin_diff = df_fin_diff./repmat(sqrt(sum(df_fin_diff.^2, 2)), 1, m);

sub3 = compute(df_fin_diff);
sub3.W1 = sub3.eigenvectors(:, 1:n);
sub3.W2 = sub3.eigenvectors(:, n+1:m);

% Plot results
opts = struct('title', 'Finite Difference Gradients');
eigenvalues(sub3.eigenvalues, sub3.e_br, opts)
eigenvectors(sub3.W1, opts)