clear variables
close all
clc

addpath 'Subspaces' 'Plotters'

%% Analytic Function Testing - EITHER THIS OR 'Data Set Testing' MUST BE COMMENTED OUT
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
m = 80; n = 2;
X_phys = data(1:m, 1:6);
f = data(1:m, 7);
df = data(1:m, 8:13);

input_ranges = [0.5, 5e-7, 2.5, 0.1, 5e-7, 0.1; ...
                  2, 5e-6, 7.5,  10, 5e-6,  10];

[M, m] = size(df);
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

% Plot results
eigenvalues(sub.eigenvalues, sub.e_br)
eigenvectors(sub.eigenvectors(:, 1:4))
subspace_errors(sub.sub_br)
sufficient_summary(X_norm*sub.eigenvectors(:, 1:2), f)