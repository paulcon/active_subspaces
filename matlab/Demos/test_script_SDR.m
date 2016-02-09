clear variables
close all
clc

addpath 'Subspaces' 'SDR' 'Plotters'

%% Test functions - Choose 1

% Linear function
m = 5; n = 2;
M = 100;
X = 2*rand(M, m) - 1;
[f, df] = test_function_1(X);
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

% Data Set
% [X, f, df] = test_function_2();
% [M, m] = size(X);
% n = 2;
% df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% Test linear_gradient_check.m

w = linear_gradient_check(X, f);

%% Test quadratic_model_check.m

gamma = ones(m, 1)/3;
[e, W] = quadratic_model_check(X, f, gamma);

eigenvalues(e)
eigenvectors(W(:, 1:2))
sufficient_summary(X*W(:, 1:2), f)