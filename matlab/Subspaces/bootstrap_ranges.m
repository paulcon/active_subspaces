function [e_br, sub_br, li_F] = bootstrap_ranges(e, W, X, f, df, weights, ssmethod, n_boot)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Use Matlab(c) bootci function capability to calculate empirical 95%
%   confidence intervals of eigenvalues and eigenvectors using the bias
%   corrected and accelerated percentile method.
%
%   Inputs:
%           e: m-by-1 array of eigenvalues
%           W: m-by-m array of eigenvectors
%           X: M-by-m array that contains data points in the input space
%           F: M-by-1 array that contains evaluations of the function
%           df: M-by-m array of gradient evaluations
%           weights: M-by-1 array of weights
%           ssmethod: function handle for computing reduced dimension
%                     subspace
%           n_boot: number of bootstrap replicates
%      
%  Outputs:
%          e_br: m-by-2 array with bootstrap eigenvalue bounds
%          sub_br: m-by-3 array of bootstrap eigenvalue ranges (first and
%                  third column) and the mean (second column)
%          li_F: (m-1)-by-1 array used in computing the ladle plot
%                partition
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup inputs
if ~isempty(X) && ~isempty(f)
    m = size(X, 2);
elseif ~isempty(df)
    m = size(df, 2);
else
    error('One of input/output pairs (X,f) or gradients (df) must not be empty')
end

% Bootstrap
e_boot = zeros(m, n_boot);
sub_dist = zeros(m-1, n_boot);
sub_det = zeros(m-1, n_boot);
for i = 1:n_boot
	[X0, f0, df0, weights0] = bootstrap_replicate(X, f, df, weights);
    [e0, W0] = ssmethod(X0, f0, df0, weights0);
    
    e_boot(:, i) = e0;
    for j = 1:m-1
        sub_dist(j, i) = norm(W(:, 1:j)'*W0(:, j+1:end), 2);
        sub_det(j, i) = det(W(:, 1:j)'*W0(:, 1:j));
    end
    
end

% Summarize Eigenvalue Basic Stats
e_br= [min(e_boot, [], 2), max(e_boot, [], 2)];

% Summarize Eigenvector Basic Stats
sub_br = [min(sub_dist, [], 2), mean(sub_dist, 2), max(sub_dist, [], 2)];

% Compute metric for Li's ladle plot
li_F = [0; sum(1 - abs(sub_det), 2)/n_boot];
li_F = li_F/sum(li_F);
    
%% Advanced bootstrap methods    
%     % Dummy indices for bootci
%     ind = ones(M,1);
%     
%     % Eigenvalue function handle
%     spec_decomp  = @(i) spectral_decomposition(df(randi(M,M,i(1)),:));
%     
%     % Eigenvector function handle
%     spec_decomp2 = @(i) spec_decomposition2(df,i,M);
% 
%     % Bootstrap eigenvalues
%     [e_br,e_dist] = bootci(n_boot,{spec_decomp,ind'},'type','stud');
%     e_br = e_br';
%     e_stat = [min(e_dist,[],1)',mean(e_dist,1)',max(e_dist,[],1)'];
%     
%     % Bootstrap eigenvectors
%     [sub_br_ci,sub_dist] = bootci(n_boot,{spec_decomp2,ind'},'type','stud');
%     lb_sub_br = zeros(m,m);
%     ub_sub_br = zeros(m,m);
%     sub_br    = zeros(m,3);
% 
%     % NOT COMPLETE, need sub_br BELOW
%     for i=1:m
%         lb_sub_br(:,i) = sub_br_ci(1,:,i)';
%         ub_sub_br(:,i) = sub_br_ci(2,:,i)';
%     end
%     
% else
%     disp('Error: Too many inputs');
% % end
% 
%     
%     
%     % Nested function to return second output of spectral_decomposition
%     function W0 = spec_decomposition2(df2,i2,M2)
%         [~,W0] = spectral_decomposition(df2(randi(M2,M2,i2(1)),:));
%     end
% 
end