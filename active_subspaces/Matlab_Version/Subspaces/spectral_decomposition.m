function [e, W] = spectral_decomposition(df, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Use the SVD to compute the eigenvectors and eigenvalues for the active
%   subspace analysis
%
%   Inputs:
%          df: M-by-m array of gradient evaluations
%          F: (optional) an ndarray of size M that contains evaluations of the function.
%          X: (optional) an ndarray of size M-by-m that contains data points in the input space.
%          fun: (optional) a specified function that outputs f(x), and df(x) the gradient vector for a data point x
%          c_index: (optional) an integer specifying which C matrix to compute, the default matrix is 0.
%          comp_flag: (optional) an integer specifying computation method: 0 for monte carlo, 1 for LG quadrature.
%          N: (optional) number of quadrature points per dimension.
%                  
%
%  Outputs:
%          e: m-by-1 array of eigenvalues
%          W: m-by-m array of eigenvectors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    F = 0;
    X = 0;
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 1
    F = varargin{1};
    X = varargin{2};
    fun = varargin{3};
    c_index = varargin{4};
    comp_flag = varargin{5};
    N = varargin{6};
    if ~isnumeric(n_boot) || rem(n_boot, 1) ~= 0 || (n_boot < 0)
        error('ERROR: n_boot must be a non-negative integer.')
    end
elseif length(varargin) == 2
    F = varargin{1};
    X = 0;
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 3
    F = varargin{1};
    X = varargin{2};
    fun = 0;
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 4
    F = varargin{1};
    X = varargin{2};
    fun = varargin{3};
    c_index = 0;
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 5
    F = varargin{1};
    X = varargin{2};
    fun = varargin{3};
    c_index = varargin{4};
    comp_flag = 0;
    N = 5;
elseif length(varargin) == 6
    F = varargin{1};
    X = varargin{2};
    fun = varargin{3};
    c_index = varargin{4};
    comp_flag = varargin{5};
     N = 5;
elseif length(varargin) == 7
    F = varargin{1};
    X = varargin{2};
    fun = varargin{3};
    c_index = varargin{4};
    comp_flag = varargin{5};
    N = varargin{6};
else
    error('ERROR: Too many inputs.')
end



% M = number of samples; m = dimension of input space;
[M,m] = size(df);
if c_index == 4
    [M,m] = size(X);
    m = m/2;
end
%Set norm tolerance
norm_tol = sqrt(eps);

% Compute the eigenvalues and eigenvectors which will ultimately form the
% active subspace.

W = zeros(m,m);
e = zeros(m,1);
C = zeros(m,m);
if c_index == 0 && comp_flag == 0
    [~, Sigma, W] = svd(df, 0);
    if M >= m
        e = (diag(Sigma).^2)/M;
    else
        e = [(diag(Sigma).^2)/M; zeros(m-M, 1)];
    end
elseif c_index == 0 && comp_flag == 1
    s = [];
    for i=1:m
        s = [s; parameter()];
    end
    order = N*ones(1,m);
    [p,w] = gaussian_quadrature(s, order);
    NN = size(w,1);

    for i=1:NN
        [~,g] = fun(p(i,:));
        C = C + (g*g')*w(i);
    end
    [W,e] = eig(C);
    [e, ind] = sort(diag(e), 'descend');
    W = W(:,ind);    
elseif c_index == 1 && comp_flag == 0
    C = X'*df+df'*X;
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2/M;
elseif c_index == 1 && comp_flag == 1
    s = [];
    for i=1:m
        s = [s; parameter()];
    end
    order = N*ones(1,m);
    [p,w] = gaussian_quadrature(s, order);
    NN = size(w,1);

    for i=1:NN
        xxx = p(i,:);
        [~,DF] = fun(xxx);
        C = C + (xxx'*DF' + DF*xxx)*w(i);
    end
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2;
elseif c_index == 2 && comp_flag == 0
    for i =1:M
       Norm = norm(df(i,:));
       if Norm < norm_tol
           df(i,:) = zeros(1,m);
       else
           df(i,:) = df(i,:)/Norm;
       end    
    end
    C = df'*df/M;
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2;
elseif c_index == 2 && comp_flag == 1
    s = [];
    for i=1:m
        s = [s; parameter()];
    end
    order = N*ones(1,m);
    [p,w] = gaussian_quadrature(s, order);
    NN = size(w,1);

    for i=1:NN
       [~,DF] =  fun(p(i,:));
       Norm = norm(DF);
       if Norm > norm_tol
          DF = DF/Norm;
          C = C + DF*DF'*w(i);
       end
    end 
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2;
    
elseif c_index == 3 && comp_flag == 0
    for i=1:M
        xxx = X(i,:);
        DF = df(i,:);
        if norm(xxx) < norm_tol
            xxx = zeros(1,m);
        else
            xxx = xxx/norm(xxx);
        end
        if norm(DF) < norm_tol
            DF = zeros(m,1);
        else
            DF = DF/norm(DF);
        end
        df(i,:) = DF';
        X(i,:) = xxx;
    end
    C = (df'*X+X'*df)/M;
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2;
elseif c_index == 3 && comp_flag == 1
    s = [];
    for i=1:m
        s = [s; parameter()];
    end
    order = N*ones(1,m);
    [p,w] = gaussian_quadrature(s, order);
    NN = size(w,1);

    for i=1:NN
        xxx = p(i,:);
        [~,DF] = fun(xxx);
        if norm(xxx) < norm_tol
            xxx = zeros(1,m);
        else
            xxx = xxx/norm(xxx);
        end
        if norm(DF) < norm_tol
            DF = zeros(m,1);
        else
            DF = DF/norm(DF);
        end
        C = C + (DF*xxx+xxx'*DF')*w(i);
    end
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2;
elseif c_index == 4 && comp_flag == 0
    for i = 1:M
        Norm = norm(X(i,1:m) - X(i,m+1:end));
        if Norm > norm_tol
            xxx = X(i,1:m);
            yyy = X(i,m+1:end);
            C = C +(xxx-yyy)'*(xxx-yyy)*(F(i)-F(i+M))^2/Norm^2;
        end
    end
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2; 
elseif c_index == 4 && comp_flag == 1
    s = [];
    for i=1:2*m
        s = [s; parameter()];
    end
    order = N*ones(1,2*m);
    [p,w] = gaussian_quadrature(s, order);
    NN = size(w,1);
    for i = 1:NN
        Norm = norm(p(i,1:m) - p(i,m+1:end));
        if Norm > norm_tol
            xxx = p(i,1:m);
            yyy = p(i,m+1:end);
            [f_x,~] = fun(xxx);
            [f_y,~] = fun(yyy);
            C = C +w(i)*(xxx-yyy)'*(xxx-yyy)*(f_x-f_y)^2/Norm^2;
        end
    end
    [~, Sigma, W] = svd(C, 0);
    e = diag(Sigma).*2; 
end
  
e(e < 0) = 0;
mult = sign(W(1, :));
mult(mult == 0) = 1;
W = W.*repmat(mult, m, 1);

end