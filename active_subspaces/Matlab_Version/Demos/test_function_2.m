function [X, f, df] = test_function_2()

data = importdata('test_data.dat');

% Set M=80 for full data set.
M = 80;
X = data(1:M, 1:6); f = data(1:M, 7); df = data(1:M, 8:13);

input_ranges = [0.5, 5e-7, 2.5, 0.1, 5e-7, 0.1; ...
                  2, 5e-6, 7.5,  10, 5e-6,  10];

[M, ~] = size(df);

for i = 1:M
    % Scale inputs so that they are on the interval [-1, 1]
    X(i, :) = 2*(X(i, :) - input_ranges(1, :))./(input_ranges(2, :) - input_ranges(1, :)) - 1;
    
    % Apply scaling to gradient.
    df(i, :) = df(i, :).*(input_ranges(2, :) - input_ranges(1, :))/2;
    
    % Normalize gradient.  Not sure if this is really important
    df(i, :) = df(i, :)/norm(df(i, :));
end

end