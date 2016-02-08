clear variables
close all
clc
%% Script configuration
% CD must be "Matlab_Port"
addpath 'ResponseSurfaces'

% Degree of polynomial basis
N=2;

% User inputs
test_reg = 'rbf'; % 'rbf' 'poly'
test_fcn = 'regression'; %'exact' 'regression' 

%% Analytic Function Testing - EITHER THIS OR 'Data Set Testing' must be commented out
% %2D Exact Quadratic Test
switch test_fcn
    case 'exact'
        m=2;

        % Generate initial grid observations for function evaluations
        [X,Y] = meshgrid([-1:0.1:1],[-1:0.1:1]);
        C0=2;
        C1=2;
        C2=0.5;
        C3=2;
        C4=1;
        C5=1;

        f =@(X,Y) C0 +  C1*Y + C2*X + C3*X.*Y + C4*X.^2 + C5*Y.^2;
        feval = f(X,Y);

        % Reshape Grid Observations
        [M2,~] = size(X);
        M=M2*M2;
        Xtrain = [reshape(X,M,1),reshape(Y,M,1)];
        fval = reshape(feval,M,1);

        % Create New Observations
        % Random points in domain
        M = 10;
        Xrand = -1 + 2*rand(M,2);

        % Linear pattern in domain
        Xlin = [(-1:0.1:1)',(-1:0.1:1)'];

        Xnew = [Xrand;Xlin];
    case 'regression'
        disp(['Runing',' ',test_fcn]);
        % 2D Regression Test
        m=2;

        % Generate initial grid observations for function evaluations
        [X,Y] = meshgrid([-1:0.1:1],[-1:0.1:1]);
        C0=2;
        C1=1;
        C2=1;
        C3=2;
        C4=-2;
        C5=2;

        f =@(X,Y) C0 +  C1*Y + C2*X + C3*X.*Y + C4*X.^2 + C5*Y.^2;
        feval = f(X,Y);

        % Reshape Grid Observations & Add Gaussian Noise
        [M2,~] = size(X);
        M=M2*M2;
        Xtrain = [reshape(X,M,1),reshape(Y,M,1)];
        fval = reshape(feval,M,1);
        % Noise
        fval = fval+ 0.5*randn(M,1);

        % Create New Observations
        % Random points in domain
        M = 10;
        Xrand = -1 + 2*rand(M,2);

        % Linear pattern in domain
        Xlin = [(-1:0.1:1)',(-1:0.1:1)'];

        Xnew = [Xrand;Xlin];
end

%% ResponseSurface Functions
switch lower(test_reg)
    case 'poly'
    % Compute Polynomial Response Surface
    [Coef,B,~,~,f_hat,r] = poly_train(Xtrain,fval,N);
    % Approximate New Function Evals
    [fpoly,dfpoly] = poly_predict(Xnew,Coef,N);

    case 'rbf'
    % Compute Radial Basis Response Surface
    [net,f_hat,r] = rbf_train(Xtrain,fval);
    % Approximate New Functino Evals
    frbf = rbf_predict(Xnew,net);
end

%% 2D Plots
switch lower(test_fcn)
    case 'exact'
        % Plot Analytic Surface
        figure(1)
        surf(X,Y,feval)
        hold on

        % Plot Analytic Contours
        figure(2)
        contour(X,Y,feval,200)
        hold on
        
        switch lower(test_reg)
            case 'poly'
                % Select Nodes of Mesh and surface plot (zero residuals)
                figure(1)
                scatter3(Xtrain(:,1),Xtrain(:,2),B*Coef,'ko');

                % Plot Resulting Approximations
                figure(1)
                scatter3(Xnew(:,1),Xnew(:,2),fpoly,'k','MarkerFaceColor','k');
                quiver3(Xnew(:,1),Xnew(:,2),fpoly,dfpoly(:,1),dfpoly(:,2),zeros(length(dfpoly),1),0.5);

                figure(2)
                quiver(Xnew(:,1),Xnew(:,2),dfpoly(:,1),dfpoly(:,2),0.5,'k');
            case 'rbf'
                % Select Nodes of Mesh and surface plot (zero residuals)
                figure(1)
                scatter3(Xtrain(:,1),Xtrain(:,2),f_hat,'ko');
                
                % Plot Resulting Approximations
                figure(1)
                scatter3(Xnew(:,1),Xnew(:,2),frbf,'k','MarkerFaceColor','k');
        end
    case 'regression'
        % Plot Training Data
        figure(1)
        scatter3(Xtrain(:,1),Xtrain(:,2),fval,'k.')
        hold on

        % Surface plot of approximation
        figure(1)
        switch lower(test_reg)
            case 'poly';
                surf(reshape(Xtrain(:,1),M2,M2),reshape(Xtrain(:,2),M2,M2),reshape(B*Coef,M2,M2));
                
                % Plot Resulting Approximations of new inputs
                scatter3(Xnew(:,1),Xnew(:,2),fpoly,'k','MarkerFaceColor','k');
                quiver3(Xnew(:,1),Xnew(:,2),fpoly,dfpoly(:,1),dfpoly(:,2),zeros(length(dfpoly),1),0.5);
                figure(2)
                quiver(Xnew(:,1),Xnew(:,2),dfpoly(:,1),dfpoly(:,2),0.5,'k');
                
                % Plot Analytic Contours
                figure(2)
                contour(X,Y,feval,200)
                hold on
                figure(2)
                quiver(Xnew(:,1),Xnew(:,2),dfpoly(:,1),dfpoly(:,2),0.5,'k');
                xlabel 'x1'
                ylabel 'x2'
                zlabel 'f(x)'
                title([test_reg,'-','Regression Approximation']);
            case 'rbf'
                surf(reshape(Xtrain(:,1),M2,M2),reshape(Xtrain(:,2),M2,M2),reshape(f_hat,M2,M2));
                
                % Plot Resulting Approximations of new inputs
                scatter3(Xnew(:,1),Xnew(:,2),frbf,'k','MarkerFaceColor','k');
        end
        
        
end
% Plot Approximation vs True Function Evaluations
figure(3)
scatter(f_hat,fval)
hold on
plot(min([f_hat;fval]):0.1:max([f_hat;fval]),min([f_hat;fval]):0.1:max([f_hat;fval]),'k-')
grid on
figure(3)
xlabel 'f_{hat}'
ylabel 'f(x)'

figure(1)
xlabel 'x1'
ylabel 'x2'
zlabel 'f(x)'
title([test_reg,'-','Regression Approximation']);