clear variables
close all
clc

% Are we going to have to do this for all the folders?
addpath 'Subspaces' 'Plotters'

%% Analytic Function Testing - EITHER THIS OR 'Data Set Testing' must be commented out
% 
% m = 5;
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

% Get data.
inputs = data(1:m, 1:6);
f = data(1:m, 7);
df = data(1:m, 8:13);

input_ranges = [0.5, 5e-7, 2.5, 0.1, 5e-7, 0.1; ...
                  2, 5e-6, 7.5,  10, 5e-6,  10];

[M, m] = size(df);
for i = 1:M
    % Scale inputs so that they are on the interval [-1, 1]
    inputs(i, :) = 2*(inputs(i, :) - input_ranges(1, :))./(input_ranges(2, :) - input_ranges(1, :)) - 1;
    
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

%% Test 'Plotters' functions
% opts = struct('title','Eigenvalues',...
%               'xlaBEl','Index',...
%               'YLabel','Eigenvalues',...
%               'XTickLabel',[]);
% opts.XTickLabel = {'rho', 'mu', 'dpdz', 'c_p', 'k', 'Pr_t'};

eigenvalues(sub.eigenvalues, sub.e_br)
eigenvectors(sub.eigenvectors(:, 1:2))
subspace_errors(sub.sub_br)
sufficient_summary(inputs*sub.W1, f)