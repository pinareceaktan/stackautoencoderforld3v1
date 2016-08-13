function [S] = transform_points(P)
m   = 0.5;
n   = 0.5;
Md  = zeros(2,1);   
Md(1) = (P(1) + m*P(2) - m*n)/(m^2 + 1);
Md(2) = m*Md(1) + n;
S = 2*Md - P;
end