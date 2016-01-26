function [y,dy] = borehole(xx)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PISTON CIRCUIT
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
% OUTPUT AND INPUT:
%
% y  = water flow rate
% xx = [rw, r, Tu, Hu, Tl, Hl, L, Kw]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scaling input from [-1,1] hypercube to the input parameter ranges
rw = (xx(1)+1)*0.5*(0.15-0.05)+0.05;
r  = (xx(2)+1)*0.5*(50000-100)+100;
Tu = (xx(3)+1)*0.5*(115600-63070)+63070;
Hu = (xx(4)+1)*0.5*(1110-990)+990;
Tl = (xx(5)+1)*0.5*(116-63.1)+63.1;
Hl = (xx(6)+1)*0.5*(820-700)+700;
L  = (xx(7)+1)*0.5*(1680-1120)+1120;
Kw = (xx(8)+1)*0.5*(12045-9855)+9855;

frac1 = 2 * pi * Tu * (Hu-Hl);

frac2a = 2*L*Tu / (log(r/rw)*rw^2*Kw);
frac2b = Tu / Tl;
frac2 = log(r/rw) * (1+frac2a+frac2b);
%Model evaluation
y = frac1 / frac2;
%Gradient evaluation
dy = [  %dy/drw
        2.*((-1).*Hl+Hu).*pi.*rw.^(-1).*Tu.*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).* ... \
        L.*rw.^(-2).*Tu.*log(r.*rw.^(-1)).^(-1)).^(-1).*log(r.*rw.^(-1)) ... \
        .^(-2)+(-2).*((-1).*Hl+Hu).*pi.*Tu.*(2.*Kw.^(-1).*L.*rw.^(-3).* ... \
        Tu.*log(r.*rw.^(-1)).^(-2)+(-4).*Kw.^(-1).*L.*rw.^(-3).*Tu.*log( ... \
        r.*rw.^(-1)).^(-1)).*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2).* ... \
        Tu.*log(r.*rw.^(-1)).^(-1)).^(-2).*log(r.*rw.^(-1)).^(-1);
        %dy/dr
        4.*((-1).*Hl+Hu).*Kw.^(-1).*L.*pi.*r.^(-1).*rw.^(-2).*Tu.^2.*(1+ ... \
        Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.*log(r.*rw.^(-1)).^(-1)) ... \
        .^(-2).*log(r.*rw.^(-1)).^(-3)+(-2).*((-1).*Hl+Hu).*pi.*r.^(-1).* ... \
        Tu.*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.*log(r.*rw.^(-1)) ... \
        .^(-1)).^(-1).*log(r.*rw.^(-1)).^(-2);
        %dy/dTu
        (-2).*((-1).*Hl+Hu).*pi.*Tu.*(Tl.^(-1)+2.*Kw.^(-1).*L.*rw.^(-2).* ... \
        log(r.*rw.^(-1)).^(-1)).*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2) ... \
        .*Tu.*log(r.*rw.^(-1)).^(-1)).^(-2).*log(r.*rw.^(-1)).^(-1)+2.*(( ... \
        -1).*Hl+Hu).*pi.*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.* ... \
        log(r.*rw.^(-1)).^(-1)).^(-1).*log(r.*rw.^(-1)).^(-1); 
        %dy/dHu
        2.*pi.*Tu.*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.*log(r.* ... \
        rw.^(-1)).^(-1)).^(-1).*log(r.*rw.^(-1)).^(-1);
        %dy/dTl
        2.*((-1).*Hl+Hu).*pi.*Tl.^(-2).*Tu.^2.*(1+Tl.^(-1).*Tu+2.*Kw.^(-1) ... \
        .*L.*rw.^(-2).*Tu.*log(r.*rw.^(-1)).^(-1)).^(-2).*log(r.*rw.^(-1)) ... \
        .^(-1);
        %dy/dHl
        (-2).*pi.*Tu.*(1+Tl.^(-1).*Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.*log( ... \
        r.*rw.^(-1)).^(-1)).^(-1).*log(r.*rw.^(-1)).^(-1);
        %dy/dL
        (-4).*((-1).*Hl+Hu).*Kw.^(-1).*pi.*rw.^(-2).*Tu.^2.*(1+Tl.^(-1).* ... \
        Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.*log(r.*rw.^(-1)).^(-1)).^(-2).* ... \
        log(r.*rw.^(-1)).^(-2);
        %dy/dKw
        4.*((-1).*Hl+Hu).*Kw.^(-2).*L.*pi.*rw.^(-2).*Tu.^2.*(1+Tl.^(-1).* ... \
        Tu+2.*Kw.^(-1).*L.*rw.^(-2).*Tu.*log(r.*rw.^(-1)).^(-1)).^(-2).* ... \
        log(r.*rw.^(-1)).^(-2);
];
dy = dy.*[0.15-0.05;50000-100;115600-63070;1110-990;116-63.1;820-700;1680-1120;12045-9855]*0.5;
end
