function [f, df] = test_function_1(X)

M = size(X, 1);

A = [1; -5; 4; 3; 3];

f = X*A;

df = repmat(A', M, 1);

end