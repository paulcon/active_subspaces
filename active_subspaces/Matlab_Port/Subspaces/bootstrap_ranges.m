function [e_br, sub_br, e_stat] = bootstrap_ranges(df, W, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Use Matlab(c) bootci function capability to calculate empirical 95%
%   confidence intervals of eigenvalues and eigenvectors using the bias
%   corrected and accelerated percentile method.
%
%   Inputs:
%          df: M-by-m array of gradient evaluations
%           W: m-by-m array of eigenvectors
%      n_boot: (optional) number of bootstrap replicates
%
%  Outputs:
%          e_br: m-by-2 array with bootstrap eigenvalue bounds
%        sub_br: m-by-3 array of bootstrap eigenvalue ranges (first and
%                third column) and the mean (second column)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if max(size(varargin)) <= 1
    
    % Check variable inputs
    if isempty(varargin)
        n_boot = 200;
    else
        n_boot = varargin{1};
    end

    % Number of gradient samples and dimension
    [M,m] = size(df);
    
%% Basic Min/Max bootstrap intervals
    % Bootstrap indices
    ind = randi(M,M,n_boot);
    
    % Bootstrap
    e_dist = zeros(m,n_boot);
    sub_dist = zeros(m-1,n_boot);
    for i=1:n_boot
        [e_dist(:,i),W0] = spectral_decomposition(df(ind(:,i),:));
        for j=1:m-1
            sub_dist(j,i) = norm(W(:,1:j)'*W0(:,j+1:end));
        end
    end
    
    % Summarize Eigenvalue Basic Stats
    e_stat = [min(e_dist,[],2),mean(e_dist,2),max(e_dist,[],2)];
    e_br= [e_stat(:,1),e_stat(:,3)];
    
    %Summarize Eigenvector Basic Stats
    sub_br = [min(sub_dist,[],2),mean(sub_dist,2),max(sub_dist,[],2)];
    
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
else
    disp('Error: Too many inputs');
end
    
    % Nested function to return second output of spectral_decomposition
    function W0 = spec_decomposition2(df2,i2,M2)
        [~,W0] = spectral_decomposition(df2(randi(M2,M2,i2(1)),:));
    end

end