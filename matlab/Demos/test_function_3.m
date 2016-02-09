function [f,df] = test_function_3(xx)

a = xx(1);
b = xx(2);
c = xx(3);

f = a^3+a*b-a*b*c;

df = [(3*a^2 -b*c + b); a*(1-c); -a*b ];

end