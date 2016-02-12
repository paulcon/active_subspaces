function [n, errbnd] = errbnd_partition(e, sub_err)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Description
%
%   Inputs:
%          X: M-by-m array that contains data points in the input space
%          f: M-by-1 array that contains evaluations of the function
%          df: M-by-m array of gradient evaluations
%          weights: M-by-1 array of weights
%
%  Outputs:
%          n: integer which gives the dimension of the active subspace
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = size(e, 1);

errbnd = zeros(m-1, 1);
for i = 1:m-1
    errbnd(i) = sqrt(sum(e(1:i)))*sub_err(i) + sqrt(sum(e(i+1:m)));
end

[~, n] = min(errbnd);

end