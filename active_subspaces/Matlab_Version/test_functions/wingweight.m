function [y,dy] = wingweight(xx)

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
% y  = wing weight
% xx = [Sw, Wfw, A, LamCaps, q, lam, tc, Nz, Wdg, Wp]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shift and scale inputs from [-1,1] hypercube to describe ranges
Lub = 0.174533; %10 degrees (in radians)
Llb = -0.174533; %-10 degrees (in radians)

Sw      = (xx(1)+1)*0.5*(200-150)+150;
Wfw     = (xx(2)+1)*0.5*(300-220)+220;
A       = (xx(3)+1)*0.5*(10-6)+6;
LamCaps = (xx(4)+1)*0.5*(Lub-Llb)+Llb;
q       = (xx(5)+1)*0.5*(45-16)+16;
lam     = (xx(6)+1)*0.5*(1-0.5)+0.5;
tc      = (xx(7)+1)*0.5*(0.18-0.08)+0.08;
Nz      = (xx(8)+1)*0.5*(6-2.5)+2.5;
Wdg     = (xx(9)+1)*0.5*(2500-1700)+1700;
Wp      = (xx(10)+1)*0.5*(0.08-0.025)+0.025;

fact1 = 0.036 * Sw^0.758 * Wfw^0.0035;
fact2 = (A / ((cos(LamCaps))^2))^0.6;
fact3 = q^0.006 * lam^0.04;
fact4 = (100*tc / cos(LamCaps))^(-0.3);
fact5 = (Nz*Wdg)^0.49;

term1 = Sw * Wp;

y = fact1*fact2*fact3*fact4*fact5 + term1;

dy = [ 
    %dy/dSw
    Wp+0.685444E-2.*lam.^0.4E-1.*q.^0.6E-2.*Sw.^(-0.242E0).*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*(tc.*sec(LamCaps)).^(-0.3E0).*(A.*sec( ... \
    LamCaps).^2).^0.6E0;
    %dy/dWfw
    0.316498E-4.*lam.^0.4E-1.*q.^0.6E-2.*Sw.^0.758E0.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^(-0.9965E0).*(tc.*sec(LamCaps)).^(-0.3E0).*(A.*sec( ... \
    LamCaps).^2).^0.6E0;
    %dy/dA
    0.542567E-2.*lam.^0.4E-1.*q.^0.6E-2.*Sw.^0.758E0.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*sec(LamCaps).^2.*(tc.*sec(LamCaps)).^( ... \
    -0.3E0).*(A.*sec(LamCaps).^2).^(-0.4E0);
    %dy/dLamCaps
    0.108513E-1.*A.*lam.^0.4E-1.*q.^0.6E-2.*Sw.^0.758E0.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*sec(LamCaps).^2.*(tc.*sec(LamCaps)).^( ... \
    -0.3E0).*(A.*sec(LamCaps).^2).^(-0.4E0).*tan(LamCaps)+( ... \
    -0.271284E-2).*lam.^0.4E-1.*q.^0.6E-2.*Sw.^0.758E0.*tc.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*sec(LamCaps).*(tc.*sec(LamCaps)).^( ... \
    -0.13E1).*(A.*sec(LamCaps).^2).^0.6E0.*tan(LamCaps);
    %dy/dq
    0.542567E-4.*lam.^0.4E-1.*q.^(-0.994E0).*Sw.^0.758E0.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*(tc.*sec(LamCaps)).^(-0.3E0).*(A.*sec( ... \
    LamCaps).^2).^0.6E0;
    %dy/dlam
    0.361712E-3.*lam.^(-0.96E0).*q.^0.6E-2.*Sw.^0.758E0.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*(tc.*sec(LamCaps)).^(-0.3E0).*(A.*sec( ... \
    LamCaps).^2).^0.6E0;
    %dy/dtc
    (-0.271284E-2).*lam.^0.4E-1.*q.^0.6E-2.*Sw.^0.758E0.*(Nz.*Wdg) ... \
    .^0.49E0.*Wfw.^0.35E-2.*sec(LamCaps).*(tc.*sec(LamCaps)).^( ... \
    -0.13E1).*(A.*sec(LamCaps).^2).^0.6E0;
    %dy/dNz
    0.443097E-2.*lam.^0.4E-1.*q.^0.6E-2.*Sw.^0.758E0.*Wdg.*(Nz.*Wdg) ... \
    .^(-0.51E0).*Wfw.^0.35E-2.*(tc.*sec(LamCaps)).^(-0.3E0).*(A.*sec( ... \
    LamCaps).^2).^0.6E0;
    %dy/dWdg
    0.443097E-2.*lam.^0.4E-1.*Nz.*q.^0.6E-2.*Sw.^0.758E0.*(Nz.*Wdg).^( ... \
    -0.51E0).*Wfw.^0.35E-2.*(tc.*sec(LamCaps)).^(-0.3E0).*(A.*sec( ... \
    LamCaps).^2).^0.6E0;
    %dy/dWp
    Sw];

dy = dy.*[50;80;4;Lub-Llb;45-16;0.5;0.10;6-2.5;800;0.08-0.025]*0.5;

end
