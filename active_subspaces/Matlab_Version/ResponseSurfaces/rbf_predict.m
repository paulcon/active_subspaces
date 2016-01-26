function [f] = rbf_predict(X, net)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Evaluate the Radial Basis Function approximation with default Guassian
%   transfer function
%
%   Inputs:
%          X: The M-by-m array of points to evaluate the RBF
%             approximation such that m is the number of dimensions
%        net: The Radial Basis Neural Network structured array
%
%   Outputs:
%          f: The M-by-1 function evaluation approximations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = sim(net,X');
f=f';