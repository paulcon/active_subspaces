function [f, df] = test_function(X)

m = size(X, 2);

A = magic(m);
A = A*A';

f = 0.5*diag(X*A*X');
df = X*A;

end