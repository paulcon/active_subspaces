function design = interval_design(a, b, N)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Obtain equally-spaced points on an interval.
%
%   Inputs:
%          a: the left endpoint of the interval
%          b: the right endpoint of the interval
%          N: the number of points in the design
%
%  Outputs:
%          design: N-by-1 array containing the design points in the
%                  interval (does not contain the endpoints)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(a, b, N+2);
design = x(2:end-1);

end