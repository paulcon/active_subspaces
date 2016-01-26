function [f, df] = analytic_function(X, varargin)

if isempty(varargin)
    QofI = 'U_avg';
elseif length(varargin) == 1
    QofI = varargin{1};
else
    error('ERROR: Too many inputs.')
end

M = size(X, 1);

mu = X(:,1);
rho = X(:,2);
dpdx = X(:,3);
eta = X(:,4);
B0 = X(:,5);

Ha = B0./sqrt(eta.*mu);
mu0 = 1;

if strcmp(QofI, 'U_avg')
    f = -dpdx.*(eta.*mu - Ha.*eta.*mu./tanh(Ha))./(mu.*B0.^2);
    
    df_dmu = -dpdx.*(sqrt(eta.*mu)./tanh(Ha) - B0./sinh(Ha).^2)./(2*B0.*mu.^2);
    df_drho = 1e-10 + (1e-8 - 1e-10)*rand(M, 1);
    df_ddpdx = -(eta.*mu - Ha.*eta.*mu./tanh(Ha))./(mu.*B0.^2);
    df_deta = -dpdx.*(2*eta.*mu - Ha.*eta.*mu./tanh(Ha) - (B0./sinh(Ha)).^2)./(2*eta.*mu.*B0.^2);
    df_dB0 = -dpdx.*(-2*eta.*mu + Ha.*eta.*mu./tanh(Ha) + (B0./sinh(Ha)).^2)./(mu.*B0.^3);
elseif strcmp(QofI, 'B_ind')
    f = dpdx.*mu0.*(B0 - 2*sqrt(eta.*mu).*tanh(Ha/2))./(2*B0.^2);
    
    df_dmu = -dpdx.*mu0.*(sqrt(eta.*mu).*sinh(Ha) - B0)./(4*mu.*(B0.*cosh(Ha/2)).^2);
    df_drho = 1e-10 + (1e-8 - 1e-10)*rand(M, 1);
    df_ddpdx = mu0.*(B0 - 2*sqrt(eta.*mu).*tanh(Ha/2))./(2*B0.^2);
    df_deta = -dpdx.*mu0.*(sqrt(eta.*mu).*sinh(Ha) - B0)./(4*eta.*(B0.*cosh(Ha/2)).^2);
    df_dB0 = -dpdx.*mu0.*(B0 + B0./cosh(Ha/2).^2 - 4*sqrt(eta.*mu).*tanh(Ha/2))./(2*B0.^3);
end

df = [df_dmu, df_drho, df_ddpdx, df_deta, df_dB0];

end