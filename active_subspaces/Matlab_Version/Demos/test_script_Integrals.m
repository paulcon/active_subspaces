clear variables
close all
clc

% Are we going to have to do this for all the folders?
addpath 'Subspaces' 'Plotters' 'Domains' 'Integrals'

%% Test functions - Choose 1

% Linear function - Used for interval_quadrature_rule.m
m = 5; n = 1;
M = 100;
X = 2*rand(M, m) - 1;
[f, df] = test_function_1(X);
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

% Data Set - Used for zonotope_quadrature_rule.m
% [X, f, df] = test_function_2();
% [M, m] = size(X);
% n = 2;
% df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% Compute active subspace

% Compute and plot active subspace using gradients approximated by
% local_linear_gradients functions.
sub = compute(df);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

%% Test interval_quadrature_rule.m

[Yp, Yw] = interval_quadrature_rule(sub.W1, 10)

%% Test zonotope_quadrature_rule.m

% [Yp, Yw] = zonotope_quadrature_rule(sub.W1, 10)

