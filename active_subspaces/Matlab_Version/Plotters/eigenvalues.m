function eigenvalues(e, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Plot the estimated eigenvalues for the acitve subspace analysis with 
%   optional bootstrap ranges.
%
%   Inputs:
%          e: m-by-1 array that contains the estimated eigenvalues
%          e_br: (optional) m-by-2 array that contains the lower and upper 
%                bounds for the estimated eigenvalues
%          opts: (optional) structure array which contain plotting options
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    e_br = [];
    opts = [];
elseif length(varargin) == 1
    if isnumeric(varargin{1})
        e_br = varargin{1};
        opts = [];
    elseif isstruct(varargin{1})
        e_br = [];
        opts = varargin{1};
    else
        error('ERROR: Inappropriate inputs passed')
    end
elseif length(varargin) == 2
    if isnumeric(varargin{1}) && isstruct(varargin{2})
       e_br = varargin{1};
       opts = varargin{2};
    elseif isstruct(varargin{1}) && isnumeric(varargin{2})
        opts = varargin{1};
        e_br = varargin{2};
    else
        error('ERROR: Inappropriate inputs passed')
    end
else
    error('ERROR: Too many inputs.')
end

m = length(e);

% Get plotting options.
opts = plot_opts(opts);

figure()

% Plot bootstrap errors if provided.
if ~isempty(e_br)
    fill([1:1:m, m:-1:1], [e_br(:, 1)', fliplr(e_br(:, 2)')], opts.err_color)
    hold on
end

% Plot eigenvalues.
semilogy(e, ...
         'markeredgecolor', 'k', ...
         'markerfacecolor', opts.color, ...
         'color', opts.color, ...
         'marker', opts.marker, ...
         'markersize', opts.markersize, ...
         'linewidth', opts.linewidth)

% Format plot.
title(opts.title, 'fontsize', opts.fontsize)

if isempty(opts.xlabel)
    xlabel('Index', 'fontsize', opts.fontsize)
else
    xlabel(opts.xlabel, 'fontsize', opts.fontsize)
end

if isempty(opts.ylabel)
    ylabel('Eigenvalues', 'fontsize', opts.fontsize)
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