function [y,dy] = robot(xx)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ROBOT ARM FUNCTION
%
% Author: Paul Diaz, Colorado School of Mines 
% Questions/Comments: Please email Paul Diaz at pdiaz@mines.edu
%
% Copyright 2016, Paul Diaz, Colorado School of Mines 
%
% THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
% FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
% derivative works, such modified software should be clearly marked.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%
% For function details and reference information, see:
% http://www.sfu.ca/~ssurjano/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% OUTPUT AND INPUTS:
%
% y = distance from the end of the arm to the origin
% xx = [theta1, theta2, theta3, theta4, L1, L2, L3, L4]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shift and scale inputs from [-1,1] hypercube to describe ranges
b = pi/2;
a = -pi/2;
theta = (xx(1:4)+1)*(b-a)*0.5+a;
L     = (xx(5:8)+1)*0.5+a;
L1 = L(1);
L2 = L(2);
L3 = L(3);
L4 = L(4);
T1 = theta(1);
T2 = theta(2);
T3 = theta(3);
T4 = theta(4);

sumu = 0;
sumv = 0;
for ii = 1:4
    Li = L(ii);
    sumtheta = 0;
    for jj = 1:ii
        thetai = theta(jj);
        sumtheta = sumtheta + thetai;
    end
    sumu = sumu + Li*cos(sumtheta);
    sumv = sumv + Li*sin(sumtheta);
end

u = sumu;
v = sumv;

y = (u^2 + v^2)^(0.5);

dy = [  %dy/dT1
        1e-6;
        %dy/dT2 
        (1/2).*(2.*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos( ... \
        T1+T2+T3+T4)).*((-1).*L2.*sin(T1+T2)+(-1).*L3.*sin(T1+T2+T3)+(-1) ... \
        .*L4.*sin(T1+T2+T3+T4))+2.*(L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.* ... \
        cos(T1+T2+T3+T4)).*(L1.*sin(T1)+L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+ ... \
        L4.*sin(T1+T2+T3+T4))).*((L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+ ... \
        T2+T3)+L4.*cos(T1+T2+T3+T4)).^2+(L1.*sin(T1)+L2.*sin(T1+T2)+L3.* ... \
        sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4)).^2).^(-1/2);
        %dy/dT3
        (1/2).*(2.*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos( ... \
        T1+T2+T3+T4)).*((-1).*L3.*sin(T1+T2+T3)+(-1).*L4.*sin(T1+T2+T3+T4) ... \
        )+2.*(L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+T3+T4)).*(L1.*sin(T1)+L2.* ... \
        sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4))).*((L1.*cos(T1) ... \
        +L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+T3+T4)).^2+(L1.* ... \
        sin(T1)+L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4)).^2) ... \
        .^(-1/2);
        %dy/dT4
        (1/2).*((-2).*L4.*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+ ... \
        L4.*cos(T1+T2+T3+T4)).*sin(T1+T2+T3+T4)+2.*L4.*cos(T1+T2+T3+T4).*( ... \
        L1.*sin(T1)+L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4)) ... \
        ).*((L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+ ... \
        T3+T4)).^2+(L1.*sin(T1)+L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin( ... \
        T1+T2+T3+T4)).^2).^(-1/2);
        %dy/dL1
        (1/2).*(2.*cos(T1).*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+ ... \
        L4.*cos(T1+T2+T3+T4))+2.*sin(T1).*(L1.*sin(T1)+L2.*sin(T1+T2)+L3.* ... \
        sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4))).*((L1.*cos(T1)+L2.*cos(T1+T2) ... \
        +L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+T3+T4)).^2+(L1.*sin(T1)+L2.*sin( ... \
        T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4)).^2).^(-1/2);
        %dy/dL2
        (1/2).*(2.*cos(T1+T2).*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+ ... \
        T3)+L4.*cos(T1+T2+T3+T4))+2.*sin(T1+T2).*(L1.*sin(T1)+L2.*sin(T1+ ... \
        T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4))).*((L1.*cos(T1)+L2.* ... \
        cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+T3+T4)).^2+(L1.*sin(T1) ... \
        +L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4)).^2).^( ... \
        -1/2);
        %dy/dL3
        (1/2).*(2.*cos(T1+T2+T3).*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+ ... \
        T2+T3)+L4.*cos(T1+T2+T3+T4))+2.*sin(T1+T2+T3).*(L1.*sin(T1)+L2.* ... \
        sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4))).*((L1.*cos(T1) ... \
        +L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+T3+T4)).^2+(L1.* ... \
        sin(T1)+L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4)).^2) ... \
        .^(-1/2);
        %dy/dL4
        (1/2).*(2.*cos(T1+T2+T3+T4).*(L1.*cos(T1)+L2.*cos(T1+T2)+L3.*cos( ... \
        T1+T2+T3)+L4.*cos(T1+T2+T3+T4))+2.*sin(T1+T2+T3+T4).*(L1.*sin(T1)+ ... \
        L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4))).*((L1.* ... \
        cos(T1)+L2.*cos(T1+T2)+L3.*cos(T1+T2+T3)+L4.*cos(T1+T2+T3+T4)).^2+ ... \
        (L1.*sin(T1)+L2.*sin(T1+T2)+L3.*sin(T1+T2+T3)+L4.*sin(T1+T2+T3+T4) ... \
        ).^2).^(-1/2);
];
dy = dy.*[2*pi;2*pi;2*pi;2*pi;1;1;1;1]*0.5;

end
