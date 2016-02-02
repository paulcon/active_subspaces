clear variables
close all
clc

addpath 'Subspaces' 'Plotters' 'Domains' 'Designs'

%% Test functions - Choose 1

% Linear function - Used to test interval_design.m
% m = 5; n = 1;
% M = 100;
% X = 2*rand(M, m) - 1;
% [f, df] = test_function_1(X);
% df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

% Data Set - Used to test maximin_design.m and gauss_hermite_design.m
[X, f, df] = test_function_2();
[M, m] = size(X);
n = 2;
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% Compute active subspace

sub = compute(df);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

[~, Y] = zonotope_vertices(sub.W1);

%% Test interval_design.m

% design = interval_design(min(Y), max(Y), 10);

%% Test maximin_design.m

design = maximin_design(Y, 30);

figure
scatter(Y(:, 1), Y(:, 2), ...
        'marker', 'o', ...
        'markerfacecolor', 'b')
hold on
scatter(design(:, 1), design(:, 2), ...
        'marker', 'o', ...
        'markerfacecolor', 'r')

%% Test gauss_hermite_design.m

design = gauss_hermite_design([3, 4]);