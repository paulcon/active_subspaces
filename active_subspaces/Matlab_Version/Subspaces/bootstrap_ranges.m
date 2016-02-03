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
%           F: (optional) an ndarray of size M that contains evaluations of the function.
%           X: (optional) an ndarray of size M-by-m that contains data points in the input space.
%     c_index: (optional) an integer specifying which C matrix to compute, the default matrix is 0.
%      
%
%  Outputs:
%          e_br: m-by-2 array with bootstrap eigenvalue bounds
%        sub_br: m-by-3 array of bootstrap eigenvalue ranges (first and
%                third column) and the mean (second column)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set varable arguements
if isempty(varargin)
    n_boot = 200;
    F = 0;
    X = 0;
    c_index = 0;
elseif length(varargin) == 1
    n_boot = varargin{1};
    F = 0;
    X = 0;
    c_index = 0;
    if ~isnumeric(n_boot) || rem(n_boot, 1) ~= 0 || (n_boot < 0)
        error('ERROR: n_boot must be a non-negative integer.')
    end
elseif length(varargin) == 2
    n_boot = varargin{1};
    F = varargin{2};
    X = 0;
    c_index = 0;
elseif length(varargin) == 3
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    c_index = 0;
elseif length(varargin) == 4
    n_boot = varargin{1};
    F = varargin{2};
    X = varargin{3};
    c_index = varargin{4};
else 
    error('ERROR: Too many inputs.')
end


% M = number of samples; m = dimension of input space;
if c_index ~= 4
    [M,m] = size(df);
else
    [M,m] = size(X);
    m = m/2;
end
    
    
    
%% Basic Min/Max bootstrap intervals
    % Bootstrap indices
    ind = randi(M,M,n_boot);
    
    % Bootstrap
    e_dist = zeros(m,n_boot);
    sub_dist = zeros(m-1,n_boot);
    for i=1:n_boot
        if c_index == 0
            [e_dist(:,i),W0] = spectral_decomposition(df(ind(:,i),:));
        elseif c_index == 1
            [e_dist(:,i),W0] = spectral_decomposition(df(ind(:,i),:),F,X(ind(:,i),:),0,c_index,0,0);
        elseif c_index == 2
            [e_dist(:,i),W0] = spectral_decomposition(df(ind(:,i),:),0,0,0,c_index,0,0);
        elseif c_index == 3
            [e_dist(:,i),W0] = spectral_decomposition(df(ind(:,i),:),F,X(ind(:,i),:),0,c_index,0,0);
        elseif c_index == 4
            Fx = F(1:M);
            Fy = F(M+1:end);
            f_x = Fx(ind(:,i));
            f_y = Fy(ind(:,i));
            F =  cat(1,f_x,f_y);
            [e_dist(:,i),W0] = spectral_decomposition(0,F,X(ind(:,i),:),0,c_index,0,0); 
            
        end
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
% else
%     disp('Error: Too many inputs');
% end

    
    
    % Nested function to return second output of spectral_decomposition
    function W0 = spec_decomposition2(df2,i2,M2)
        [~,W0] = spectral_decomposition(df2(randi(M2,M2,i2(1)),:));
    end

end