clear variables
close all
clc

addpath '..Subspaces' '..Plotters' '..ResponseSurfaces' '../../../tutorials/test_functions/borehole'

%% c_index = 0 MONTE CARLO
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df;
end
n_boot = 200;
c_index = 0;
comp_flag = 0;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 0 LG QUADRATURE
% non-linear test function
tic;
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
parfor i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df;
end
n_boot = 200;
c_index = 0;
comp_flag = 1;
N = 15;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
toc
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 1 MONTE CARLO
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df;
end
n_boot = 200;
c_index = 1;
comp_flag = 0;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 LG QUADRATURE
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df';
end
n_boot = 200;
c_index = 1;
comp_flag = 1;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 2 MONTE CARLO
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df;
end
n_boot = 200;
c_index = 2;
comp_flag = 0;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 2 LG QUADRATURE
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df';
end
n_boot = 200;
c_index = 2;
comp_flag = 1;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 3 MONTE CARLO
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df;
end
n_boot = 200;
c_index = 3;
comp_flag = 0;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 3 LG QUADRATURE
% non-linear test function
fun =  @borehole;
m = 8; n = 1;
M = 10000;
X = 2*rand(M, m) - 1;
% Generate a data set 
F = zeros(M,1);
DF = zeros(M,m);
for i = 1:M
    [f, df] = fun(X(i,:));
    F(i) = f; DF(i,:) = df';
end
n_boot = 200;
c_index = 3;
comp_flag = 1;
N = 10;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%%  for c_index = 4 MONTE CARLO
fun =  @test_function_3;
m = 3; 
n = 1;
M = 10000;
X = 2*rand(M, 2*m) - 1;
f_x = zeros(M,1); f_y = zeros(M,1);
for i = 1:M
    [f_x(i), ~] = fun(X(i,1:m));
    [f_y(i), ~] = fun(X(i,m+1:end));
end
F = cat(1,f_x,f_y);
n_boot = 500;
c_index = 4;
comp_flag = 0; % 0 for MC
N = 10;
DF = 1;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
X0 = cat(1,X(:,1:m),X(:,m+1:end));

% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)


%% for c_index = 4 LG Quadrature 
fun =  @test_function_3;
m = 3; 
n = 1;
M = 10000;
X = 2*rand(M, 2*m) - 1;
f_x = zeros(M,1); f_y = zeros(M,1);
for i = 1:M
    [f_x(i), ~] = fun(X(i,1:m));
    [f_y(i), ~] = fun(X(i,m+1:end));
end
F = cat(1,f_x,f_y);
n_boot = 500;
c_index = 4;
comp_flag = 1; % 1  LG Quadrature 
N = 2;
DF = 0;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
X0 = cat(1,X(:,1:m),X(:,m+1:end));
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)


