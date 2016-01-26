function opts = plot_opts(opts_custom)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Set plotting options.
%
%   Inputs:
%          opts_custom: structure which can be used to override the default 
%                       plotting options shown below
%
%  Outputs:
%          opts: structure containing plotting options
%                        
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts = struct('color','b',...
              'err_color',[0.8 0.8 0.8],...
              'marker', 'o', ...
              'markersize', 10, ...
              'linewidth', 2, ...
              'title', '', ...
              'xlabel', '', ...
              'ylabel', '', ...
              'fontsize', 12, ...
              'xticklabel', []);

if ~isempty(opts_custom)
    all_fields = fieldnames(opts);
    for field = fieldnames(opts_custom)'
        tf_fieldmatch = strcmpi(all_fields, field);
        if any(tf_fieldmatch)
            opts = setfield(opts, all_fields{tf_fieldmatch}, getfield(opts_custom, field{:}));
        else
            error(['ERROR: Unexpected field name -> ' field])
        end
    end
end


end