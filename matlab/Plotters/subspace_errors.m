function subspace_errors(sub_br, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Plot the estimated subspace errors for the active subspace analysis
%   with bootstrap ranges.
%
%   Inputs:
%          sub_br: m-by-3 array of bootstrap eigenvalue ranges (first and
%                  third column) and the mean (second column)
%          opts: (optional) structure array which contain plotting options
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    opts = [];
elseif length(varargin) == 1
    opts = varargin{1};
    if ~isstruct(opts)
        error('ERROR: Inappropriate inputs passed')
    end
else
    error('ERROR: Too many inputs.')
end

m = size(sub_br, 1);

% Get plotting options.
opts = plot_opts(opts);

figure()

% Plot bootstrap errors.
if max(abs(sub_br(:,3) - sub_br(:,1))) > 1e-7
    fill([1:1:m, m:-1:1], [sub_br(:,1)', fliplr(sub_br(:,3)')], opts.err_color)
    hold on
end

% Plog subspace errors.
semilogy(sub_br(:,2), ...
         'markeredgecolor', 'k', ...
         'markerfacecolor', opts.color, ...
         'color', opts.color, ...
         'marker', opts.marker, ...
         'markersize', opts.markersize, ...
         'linewidth', opts.linewidth)
     
% Format plot.
title(opts.title, 'fontsize', opts.fontsize)

if isempty(opts.xlabel)
    xlabel('Subspace dimension', 'fontsize', opts.fontsize)
else
    xlabel(opts.xlabel, 'fontsize', opts.fontsize)
end

if isempty(opts.ylabel)
    ylabel('Subspace distance', 'fontsize', opts.fontsize)
else
    ylabel(opts.ylabel, 'fontsize', opts.fontsize)
end

set(gca, ...
    'XLim', [1, m], ...
    'XTick', 1:m, ...
    'XScale', 'Linear', ...
    'YScale', 'Log', ...
    'fontsize', opts.fontsize)

end