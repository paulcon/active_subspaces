function sufficient_summary(Y, f, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Make 1D and 2D sufficient summary plots with the data.
%
%   Inputs:
%          Y: M-by-(1 or 2) array that contains the values of the
%             predictors for the sufficient summary plot
%          f: M-by-1 array that contains the corresponding responses
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

n = size(Y, 2);
if n > 2
    error('ERROR: sufficient summary plots cannot be made in more than 2 dimensions.')
end

% Get plotting options.
opts = plot_opts(opts);

figure()

% Make 1D sufficient summary plot.
scatter(Y(:,1), f, ...
     'markeredgecolor', 'none', ...
     'markerfacecolor', opts.color, ...
     'marker', opts.marker, ...
     'sizedata', 6*opts.markersize, ...
     'linewidth', opts.linewidth)

% Format plot
if isempty(opts.title)
    title('Output', 'fontsize', opts.fontsize)
else
    title(opts.title, 'fontsize', opts.fontsize)
end
xlabel('Active Variable 1', 'fontsize', opts.fontsize)
ylabel(opts.ylabel, 'fontsize', opts.fontsize)
set(gca, 'fontsize', opts.fontsize)
grid('on')

if n == 2
    figure()
    
    % Make 2D sufficient summary plot.
    scatter(Y(:,1), Y(:,2), 'filled', ...
            'cdata', f, ....
            'marker', opts.marker, ...
            'sizedata', 6*opts.markersize, ...
            'linewidth', opts.linewidth)

    % Format plot
    if isempty(opts.title)
        title('Output', 'fontsize', opts.fontsize)
    else
        title(opts.title, 'fontsize', opts.fontsize)
    end
    xlabel('Active Variable 1', 'fontsize', opts.fontsize)
    ylabel('Active Variable 2', 'fontsize', opts.fontsize)
    set(gca, 'fontsize', opts.fontsize)
    grid('on')
    colorbar()
    colormap('jet')
end

end