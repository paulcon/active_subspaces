clear variables
close all
clc
load test_data.mat

tests_passed = false(7, 1);

[M, m] = size(X);
[f, df] = test_function(X);
df = df./repmat(sqrt(sum(df.^2, 2)), 1, m);

%% Test Subspaces

restoredefaultpath
addpath 'Subspaces'

failed_subspaces = false;
try
    sub_test = compute(df);
    if (size(sub.W1, 2) ~= size(sub_test.W1, 2)) || (abs(1 - norm(sub.W1'*sub_test.W1)) > sqrt(eps))
        failed_subspaces = true;
    end
catch
    failed_subspaces = true;
end

if failed_subspaces
    disp('Subspaces testing unsuccessful. Error in computing active subspace.')
else
    disp('Subspaces testing successful.')
    tests_passed(1) = true;
end

%% Test Gradients

restoredefaultpath
addpath 'Subspaces' 'Gradients'

failed_loc_lin = false;
try
    loc_lin_err = 1;
    for i = 1:5
        df_loc_lin = local_linear_gradients(X, f);
        df_loc_lin = df_loc_lin./repmat(sqrt(sum(df_loc_lin.^2, 2)), 1, m);
        sub_loc_lin = compute(df_loc_lin);
        temp = norm(sub.W1'*sub_loc_lin.W2);
        if temp < loc_lin_err
            loc_lin_err = temp;
        end
    end
    if loc_lin_err > 0.2
        failed_loc_lin = true;
    end
catch
    failed_loc_lin = true;
end

failed_fin_diff = false;
try
    fun = @(X) test_function(X); h = 1e-8;
    df_fin_diff = finite_difference_gradients(X, fun, h);
    df_fin_diff = df_fin_diff./repmat(sqrt(sum(df_fin_diff.^2, 2)), 1, m);
    sub_fin_diff = compute(df_fin_diff);
    if norm(sub.W1'*sub_fin_diff.W2) > sqrt(eps)
        failed_fin_diff = true;
    end
catch
    failed_fin_diff = true;
end

clear df_loc_lin df_fin_diff sub_loc_lin sub_fin_diff

if failed_loc_lin
    disp('Gradients testing unsuccessful. Error in local linear approximation.')
elseif failed_fin_diff
    disp('Gradients testing unsuccessful. Error in finite differences.')
else
    disp('Gradients testing successful.')
    tests_passed(2) = true;
end

%% Test Domains

restoredefaultpath
addpath 'Domains'

n = 2;
W1 = sub.eigenvectors(:, 1:n); W2 = sub.eigenvectors(:, n+1:m);

failed_zonotope_verts = false;
try
    [X_vert, Y_vert] = zonotope_vertices(W1);
    if (size(X_vert, 1) ~= size(Y_vert, 1)) || norm(X_vert*W1 - Y_vert) > sqrt(eps)
        failed_zonotope_verts = true;
    end
catch
    failed_zonotope_verts = true;
end

nz = 10;
Y = X*W1; Z = zeros(M, m-n, nz);
failed_hit_and_run = false; failed_rejection_sampling = false; failed_random_walk = false;
for i = 1:M
    y = Y(i, :);
    y2 = repmat(y, nz, 1);
    
    try
        Z_hit_and_run = hit_and_run_z(nz, y', W1, W2);
        X_hit_and_run = y2*W1' + Z_hit_and_run*W2';
        hit_and_run_err = max(sum(abs(y2 - X_hit_and_run*W1).^2, 2));
        if any(any(abs(X_hit_and_run) > 1)) || (hit_and_run_err > eps)
            failed_hit_and_run = true;
        end
        
        Z(i, :, :) = reshape(Z_hit_and_run', 1, m-n, nz);
    catch
        failed_hit_and_run = true;
    end
    
    try
        Z_rejection_sampling = rejection_sampling_z(nz, y', W1, W2);
        X_rejection_sampling = y2*W1' + Z_rejection_sampling*W2';
        rejection_sampling_err = max(sum(abs(y2 - X_rejection_sampling*W1).^2, 2));
        if any(any(abs(X_rejection_sampling) > 1)) || (rejection_sampling_err > eps)
            failed_rejection_sampling = true;
        end
    catch
        failed_rejection_sampling = true;
    end
    
    try
        Z_random_walk = random_walk_z(nz, y', W1, W2);
        X_random_walk = y2*W1' + Z_random_walk*W2';
        random_walk_err = max(sum(abs(y2 - X_random_walk*W1).^2, 2));
        if any(any(abs(X_random_walk) > 1)) || (random_walk_err > eps)
            failed_random_walk = true;
        end
    catch
        failed_random_walk = true;
    end
end

[X0, ind] = rotate_x(Y, Z, sub.eigenvectors);
if any(any(abs(X0) > 1)) || (max(sum((Y(ind, :) - X0*W1).^2, 2)) > eps)
    failed_rotate_x = true;
else
    failed_rotate_x = false;
end

clear Z X0 ind 

if failed_zonotope_verts
    disp('Domains testing unsuccessful. Error in finding zonotope vertices.')
elseif failed_hit_and_run
    disp('Domains testing unsuccessful. Error in hit and run method.')
elseif failed_rejection_sampling
    disp('Domains testing unsuccessful. Error in rejection sampling method.')
elseif failed_random_walk
    disp('Domains testing unsuccessful. Error in random walk method.')
elseif failed_rotate_x
    disp('Domains testing unsuccessful. Error in rotate x method.')
else
    disp('Domains testing successful.')
    tests_passed(3) = true;
end

%% Test Designs

restoredefaultpath
addpath 'Domains' 'Designs'

failed_design_interval = false;
try
    [~, vert_1D] = zonotope_vertices(sub.eigenvectors(:, 1));
    design_interval = interval_design(min(vert_1D), max(vert_1D), 10);
    if any(design_interval < vert_1D(1)) || any(design_interval > vert_1D(2)) || issorted(-design_interval)
        failed_design_interval = true;
    end
catch
    failed_design_interval = true;
end

failed_design_maximin = false;
try 
    N_maximin = 30;
    [~, vert_2D] = zonotope_vertices(sub.eigenvectors(:, 1:2));
    design_maximin = maximin_design(vert_2D, N_maximin);
    if any(size(design_maximin) ~= [N_maximin, 2])
        failed_design_maximin = true;
    end
catch
    failed_design_maximin = true;
end

failed_design_gauss_hermite = false;
try
    N_gh = [3, 4];
    design_gauss_hermite = gauss_hermite_design(N_gh);
    if any(size(design_gauss_hermite) ~= [prod(N_gh), length(N_gh)])
        failed_design_gauss_hermite = true;
    end
catch
    failed_design_gauss_hermite = true;
end

clear design_interval design_maximin design_gauss_hermite

if failed_design_interval
    disp('Designs testing unsuccessful. Error in interval design.')
elseif failed_design_maximin
    disp('Designs testing unsuccessful. Error in maximin design.')
elseif failed_design_gauss_hermite
    disp('Designs testing unsuccessful. Error in Gauss-Hermite design.')
else
    disp('Designs testing successful.')
    tests_passed(4) = true;
end

%% Test Integrals

restoredefaultpath
addpath 'Integrals'

N_quad = 10;

failed_interval_quad = false;
try
    [Yp_interval, Yw_interval] = interval_quadrature_rule(sub.eigenvectors(:, 1), N_quad);
    if any(size(Yp_interval) ~= size(Yw_interval)) || any(Yw_interval > 1)
        failed_interval_quad = true;
    end
catch
    failed_interval_quad = true;
end

failed_zonotope_quad = false;
try
    [Yp_zonotope, Yw_zonotope] = zonotope_quadrature_rule(sub.eigenvectors(:, 1:2), N_quad);
    if (size(Yp_zonotope, 1) ~= size(Yw_zonotope, 1)) || any(Yw_zonotope > 1)
        failed_interval_quad = true;
    end
catch
    failed_zonotope_quad = true;
end

clear Yp_interval Yw_interval Yp_zonotope Yw_zonotope

if failed_interval_quad
    disp('Integrals testing unsuccessful. Error in interval quadrature construction.')
elseif failed_zonotope_quad
    disp('Integrals testing unsuccessful. Error in zonotope quadrature construction.')
else
    disp('Integrals testing successful.')
    tests_passed(5) = true;
end

%% Test SDR

restoredefaultpath
addpath 'SDR'

failed_lin_grad = false;
try
    [w, w_br] = linear_gradient_check(X, f);
    if any(w < w_br(:, 1)) || any(w > w_br(:, 2))
        failed_lin_grad = true;
    end
catch
    failed_lin_grad = true;
end

failed_quad_model = false;
try
    gamma = ones(m, 1)/3;
    [e, W] = quadratic_model_check(X, f, gamma);
    if any(e < 0) || any(size(W) ~= [m, m])
        failed_quad_model = true;
    end
catch
    failed_quad_model = true;
end

if failed_lin_grad
    disp('SDR testing unsuccessful. Error in linear gradient check.')
elseif failed_quad_model
    disp('SDR testing unsuccessful. Error in quadratic model check.')
else
    disp('SDR testing successful.')
    tests_passed(6) = true;
end

%% Test Response Surfaces
restoredefaultpath
addpath 'ResponseSurfaces'

% Test polynomial training/predicting
    % Generate initial grid observations for function evaluations
    [X,Y] = meshgrid(linspace(-1,1,100),linspace(-1,1,100));

    % Reshape Grid Observations
    [M2,~] = size(X);
    M=M2*M2;
    Xtrain = [reshape(X,M,1),reshape(Y,M,1)];
    
    % Run test function
    fval = test_function(Xtrain);
    
    % Create New Observations
    M = 50;
    Xrand = -1 + 2*rand(M,2);
    
    failed_poly_train = false; % Innocent until proven guilty I suppose
    try 
        % Compute Quadractic Polynomial Response Surface
        N=2;
        [Coef,B,~,~,f_hat,r] = poly_train(Xtrain,fval,N);
        train_err = norm(B*Coef-fval,'inf');
        if train_err >= 1e-10
            failed_poly_train = true;
        end
    catch
        failed_poly_train = true;
    end
    
    failed_poly_predict = false;
    try 
        % Approximate New Function Evals
        [fpoly,dfpoly] = poly_predict(Xrand,Coef,N);
        ftrue = test_function(Xrand);
        predict_err = norm(fpoly-ftrue,'inf');
        if predict_err >= 1e-10
            failed_poly_predict = true;
        end
    catch
        failed_poly_predict = true;
    end
    
if failed_poly_train
    disp('Polynomial Response Surface testing unsuccessful. Error in polynomial train.')
    fprintf('\t Polynomial Train Residual Inf. Norm:\t %1.16f\n',train_err);
elseif failed_poly_predict
    disp('Polynomial Response Surface testing unsuccessful. Error in polynomial predict.')
    fprintf('\t Polynomial Predict Residual Inf. Norm:\t %1.16f\n',predict_err);
else
    disp('Polynomial Response Surface testing successful:')
    fprintf('\t Polynomial Train Residual Inf. Norm:\t %1.16f\n',train_err);
    fprintf('\t Polynomial Predict Residual Inf. Norm:\t %1.16f\n',predict_err);
    tests_passed(7) = true;
end
%% Testing complete
disp(['TESTING COMPLETE: ' num2str(sum(tests_passed)) ' out of ' num2str(length(tests_passed)) ' tests passed successfully.'])
clear all