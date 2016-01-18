function [Vm,dVm] = otlcircuit(xx)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% OTL CIRCUIT FUNCTION
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
% Vm = midpoint voltage
% xx = [Rb1, Rb2, Rf, Rc1, Rc2, beta]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Rb1 in [50, 150]	resistance b1 (K-Ohms)
%Rb2 in [25, 70]	resistance b2 (K-Ohms)
%Rf in [0.5, 3]	resistance f (K-Ohms)
%Rc1 in [1.2, 2.5]	resistance c1 (K-Ohms)
%Rc2 in [0.25, 1.2]   	resistance c2 (K-Ohms)
%beta in [50, 300]	current gain (Amperes)
% Scaling input from [-1,1] hypercube to input parameter ranges
Rb1  = (xx(1)+1)*0.5*(150-50)+50;
Rb2  = (xx(2)+1)*0.5*(70-25)+25;
Rf   = (xx(3)+1)*0.5*(3-0.5)+0.5;
Rc1  = (xx(4)+1)*0.5*(2.5-1.2)+1.2;
Rc2  = (xx(5)+1)*0.5*(1.2 - 0.25)+0.25;
beta = (xx(6)+1)*0.5*(300-50)+50;
%Model evaluation
Vb1 = 12*Rb2 / (Rb1+Rb2);
term1a = (Vb1+0.74) * beta * (Rc2+9);
term1b = beta*(Rc2+9) + Rf;
term1 = term1a / term1b;

term2a = 11.35 * Rf;
term2b = beta*(Rc2+9) + Rf;
term2 = term2a / term2b;

term3a = 0.74 * Rf * beta * (Rc2+9);
term3b = (beta*(Rc2+9)+Rf) * Rc1;
term3 = term3a / term3b;

Vm = term1 + term2 + term3;
%Gradient evaluation

dVm = [ 
        (-12*Rb2*beta*(Rc2+9)*(beta*(Rc2+9)+Rf)^(-1)*(Rb1+Rb2)^(-2)); 
        12*beta*Rb1*(Rc2+9)*((Rb1+Rb2)^2*(beta*(Rc2+9)+Rf))^(-1);  
        beta*(beta*(59.94+(13.32+0.74*Rc2)*Rc2)+Rc1*(95.49+Rc2*(10.61-Vb1)-9*Vb1))*(Rc1*(beta*(Rc2+9)+Rf)^2)^(-1);
        -0.74*beta*(Rc2+9)*Rf*(Rc1^2*(beta*(Rc2+9)+Rf))^(-1); 
        beta*Rf*(-10.61*Rb1*Rc1+1.39*Rb2*Rc1+0.74*Rb1*Rf+0.74*Rb2*Rf)*((Rb1+Rb2)*Rc1*(beta*(Rc2+9)+Rf)^2  )^(-1);
        Rf*((6.66+0.74*Rc2)*Rf+Rc1*(-95.49+9*Vb1+Rc2*(-10.61+Vb1)))*(Rc1*(beta*(Rc2+9)+Rf)^2 )^(-1);
        
        ];
%scaling from y \in [-1,1] to x \in [a,b] via chain rule
dVm = dVm.*[150-50;70-25;3-.5;2.5-1.2;1.2-.25;300-50]*0.5;

end
