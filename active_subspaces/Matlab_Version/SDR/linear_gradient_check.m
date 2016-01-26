function w = linear_gradient_check(X, f, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Use the normalized gradient of a global linear model to define the
%   active subspace.
%
%   Inputs:
%          X: M-by-m array containing sample points in the input space
%          f: M-by-1 array containing simulation evalutions corresponding
%             to the input points in X
%          n_boot: (optional) integer which gives the number of bootstrap
%                  replicates to use when computing bootstrap ranges.  If
%                  n_boot=0, then bootstrapping is not performed
%                  Default: 1000
%          in_label: (optional) 1-by-m cell array containing labels for
%                    the input parameters
%          out_label: (optional) string containing the label for the
%                     simulation output
%
%  Outputs:
%          w: m-by-1 array containing the normalized gradient of the global
%             linear model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath 'Plotters'

% Set value for number of bootstrap replicants.
if isempty(varargin)
    n_boot = 1000;
    opts = struct('title', '', 'xticklabel', []);
elseif length(varargin) <= 3
    n_boot = 1000;
    opts = struct('title', '', 'xticklabel', []);
    for i = 1:length(varargin)
        if isnumeric(varargin{i})
            n_boot = varargin{i};
            if (numel(n_boot) ~= 1) || rem(n_boot, 1) ~= 0 || (n_boot < 0)
                error('ERROR: n_boot must be a non-negative integer.')
            end
        elseif iscell(varargin{i})
            opts.xticklabel = varargin{i};
        elseif ischar(varargin{i})
            opts.title = varargin{i};
        else
            error('ERROR: Inappropriate inputs passed.')
        end
    end
else
    error('ERROR: Too many inputs.')
end

[M, m] = size(X);

% Approximate gradient using a global linear model.
w = [ones(M, 1), X]\f;
w = w(2:end)/norm(w(2:end));

% Perform bootstrapping on w.
ind = randi(M, M, n_boot);
w_boot = zeros(m, n_boot);
for i = 1:n_boot
    w_temp = [ones(M, 1), X(ind(:, i), :)]\f(ind(:, i));
    w_boot(:, i) = w_temp(2:end)/norm(w_temp(2:end));
end
w_br = [min(w_boot, [], 2), max(w_boot, [], 2)];

% Make sufficient summary plot.
sufficient_summary(X*w, f, opts)

% Plot gradient.
eigenvectors(w, w_br, opts)

end