clear variables
close all
clc

addpath '../Subspaces' '../Plotters' '../ResponseSurfaces' '../test_functions'

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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('borehole_C0_MC')
%%
load('borehole_C0_MC')
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
N = 6;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('borehole_C0_LGQ')
toc
%%
load('borehole_C0_LGQ')
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('borehole_C1_MC')
%% Plot results
load('borehole_C1_MC')
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('borehole_C1_LGW')
%% Plot results
load('borehole_C1_LGW')
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('borehole_C2_MC')
%%
load('borehole_C2_MC')
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('borehole_C2_LGW')
%% Plot results
load('borehole_C2_LGW')
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('borehole_C3_MC')
%% Plot results
load('borehole_C3_MC')
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('borehole_C3_LGW')
%% Plot results
load('borehole_C3_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%%  for c_index = 4 MONTE CARLO
fun =  @borehole;
m = 8; 
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
save('borehole_C4_MC')
%%
% Plot results
load('borehole_C4_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)


%%%%%%%%%%%%%%%% OTL_CIRCUIT %%%%%%%%%%%%%%%%%%%%%
%% c_index = 0 MONTE CARLO
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('otlcircuit_C0_MC')
%%
load('otlcircuit_C0_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 0 LG QUADRATURE
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
comp_flag = 1;
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('otlcircuit_C0_LGQ')
%%
load('otlcircuit_C0_LGQ')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 MONTE CARLO
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('otlcircuit_C1_MC')
%% Plot results
load('otlcircuit_C1_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 LG QUADRATURE
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('otlcircuit_C1_LGW')
%% Plot results
load('otlcircuit_C1_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 2 MONTE CARLO
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('otlcircuit_C2_MC')
%%
load('otlcircuit_C2_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 2 LG QUADRATURE
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('otlcircuit_C2_LGW')
%% Plot results
load('otlcircuit_C2_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 3 MONTE CARLO
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('otlcircuit_C3_MC')
%% Plot results
load('otlcircuit_C3_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 3 LG QUADRATURE
% non-linear test function
fun =  @otlcircuit;
m = 6; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('otlcircuit_C3_LGW')
%% Plot results
load('otlcircuit_C3_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%%  for c_index = 4 MONTE CARLO
fun =  @otlcircuit;
m = 6; 
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
save('otlcircuit_C4_MC')
%%
% Plot results
load('otlcircuit_C4_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)

%%%%%%%%%%%%%%%% piston %%%%%%%%%%%%%%%%%%%%%
%% c_index = 0 MONTE CARLO
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('piston_C0_MC')
%%
load('piston_C0_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 0 LG QUADRATURE
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
comp_flag = 1;
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('piston_C0_LGQ')
%%
load('piston_C0_LGQ')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 MONTE CARLO
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('piston_C1_MC')
%% Plot results
load('piston_C1_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 LG QUADRATURE
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('piston_C1_LGW')
%% Plot results
load('piston_C1_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 2 MONTE CARLO
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('piston_C2_MC')
%%
load('piston_C2_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 2 LG QUADRATURE
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('piston_C2_LGW')
%% Plot results
load('piston_C2_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 3 MONTE CARLO
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('piston_C3_MC')
%% Plot results
load('piston_C3_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 3 LG QUADRATURE
% non-linear test function
fun =  @piston;
m = 7; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('piston_C3_LGW')
%% Plot results
load('piston_C3_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%%  for c_index = 4 MONTE CARLO
fun =  @piston;
m = 7; 
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
save('piston_C4_MC')
%%
% Plot results
load('piston_C4_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)

%%%%%%%%%%%%%%%% robot %%%%%%%%%%%%%%%%%%%%%
%% c_index = 0 MONTE CARLO
% non-linear test function
fun =  @robot;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('robot_C0_MC')
%%
load('robot_C0_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 0 LG QUADRATURE
% non-linear test function
fun =  @robot;
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
comp_flag = 1;
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('robot_C0_LGQ')
%%
load('robot_C0_LGQ')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 MONTE CARLO
% non-linear test function
fun =  @robot;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('robot_C1_MC')
%% Plot results
load('robot_C1_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 LG QUADRATURE
% non-linear test function
fun =  @robot;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('robot_C1_LGW')
%% Plot results
load('robot_C1_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 2 MONTE CARLO
% non-linear test function
fun =  @robot;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('robot_C2_MC')
%%
load('robot_C2_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 2 LG QUADRATURE
% non-linear test function
fun =  @robot;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('robot_C2_LGW')
%% Plot results
load('robot_C2_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 3 MONTE CARLO
% non-linear test function
fun =  @robot;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('robot_C3_MC')
%% Plot results
load('robot_C3_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 3 LG QUADRATURE
% non-linear test function
fun =  @robot;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('robot_C3_LGW')
%% Plot results
load('robot_C3_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%%  for c_index = 4 MONTE CARLO
fun =  @robot;
m = 8; 
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
save('robot_C4_MC')
%%
% Plot results
load('robot_C4_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)

%%%%%%%%%%%%%%%% wingweight %%%%%%%%%%%%%%%%%%%%%
%% c_index = 0 MONTE CARLO
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('wingweight_C0_MC')
%%
load('wingweight_C0_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 0 LG QUADRATURE
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
comp_flag = 1;
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('wingweight_C0_LGQ')
%%
load('wingweight_C0_LGQ')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 MONTE CARLO
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('wingweight_C1_MC')
%% Plot results
load('wingweight_C1_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 1 LG QUADRATURE
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 2;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('wingweight_C1_LGW')
%% Plot results
load('wingweight_C1_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 2 MONTE CARLO
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);
save('wingweight_C2_MC')
%%
load('wingweight_C2_MC')
% Plot results
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 2 LG QUADRATURE
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('wingweight_C2_LGW')
%% Plot results
load('wingweight_C2_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%% c_index = 3 MONTE CARLO
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 4;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('wingweight_C3_MC')
%% Plot results
load('wingweight_C3_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)
%% c_index = 3 LG QUADRATURE
% non-linear test function
fun =  @wingweight;
m = 10; n = 1;
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
N = 3;
% Compute active subspace
sub = compute(DF,n_boot,F,X,fun,c_index,comp_flag,N);
sub.W1 = sub.eigenvectors(:, 1:n);
sub.W2 = sub.eigenvectors(:, n+1:m);

save('wingweight_C3_LGW')
%% Plot results
load('wingweight_C3_LGW')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X*sub.eigenvectors(:, 1:2), F)

%%  for c_index = 4 MONTE CARLO
fun =  @wingweight;
m = 10; 
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
save('wingweight_C4_MC')
%%
% Plot results
load('wingweight_C4_MC')
if comp_flag == 0
eigenvalues(sub.eigenvalues, sub.e_br)
subspace_errors(sub.sub_br)
end
eigenvectors(sub.W1)
sufficient_summary(X0*sub.eigenvectors(:, 1:2), F)
