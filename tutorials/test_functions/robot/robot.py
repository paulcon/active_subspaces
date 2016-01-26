from __future__ import division
import numpy as np

def fun(xx):
##########################################################################
#
# ROBOT ARM FUNCTION
#
# Author: Paul Diaz, Colorado School of Mines 
# Questions/Comments: Please email Paul Diaz at pdiaz@mines.edu
#
# Copyright 2016, Paul Diaz, Colorado School of Mines 
#
# THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
# FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
# derivative works, such modified software should be clearly marked.
# Additionally, this program is free software; you can redistribute it 
# and/or modify it under the terms of the GNU General Public License as 
# published by the Free Software Foundation; version 2.0 of the License. 
# Accordingly, this program is distributed in the hope that it will be 
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# For function details and reference information, see:
# http://www.sfu.ca/~ssurjano/
#
##########################################################################
#
# OUTPUT AND INPUTS:
#
# y = distance from the end of the arm to the origin
# xx = [theta1, theta2, theta3, theta4, L1, L2, L3, L4]
#
#########################################################################
# Shift and scale inputs from [-1,1] hypercube to describe ranges
    pi = np.pi
    b = pi/2
    a = -pi/2
    theta = (xx[0:4]+1)*(b-a)*0.5+a
    L     = (xx[4:8]+1)*0.5+a
    L1 = L[0]
    L2 = L[1]
    L3 = L[2]
    L4 = L[3]
    T1 = theta[0]
    T2 = theta[1]
    T3 = theta[2]
    T4 = theta[3]
    
    u = L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4)
    v = L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)
    f = (u**2 + v**2)**(0.5);
    
    out = np.array([ 
        #(1/2)*(2*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(  \
        #T1+T2+T3+T4))*((-1)*L1*np.sin(T1)+(-1)*L2*np.sin(T1+T2)+(-1)*L3*  \
        #np.sin(T1+T2+T3)+(-1)*L4*np.sin(T1+T2+T3+T4))+2*(L1*np.cos(T1)+L2*np.cos(  \
        #T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))*(L1*np.sin(T1)+L2*  \
        #np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)))*((L1*np.cos(T1)  \
        #+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+(L1*  \
        #np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))**2)  \
        #**(-1/2),
        1e-12,
        (1/2)*(2*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(  \
        T1+T2+T3+T4))*((-1)*L2*np.sin(T1+T2)+(-1)*L3*np.sin(T1+T2+T3)+(-1)  \
        *L4*np.sin(T1+T2+T3+T4))+2*(L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*  \
        np.cos(T1+T2+T3+T4))*(L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+  \
        L4*np.sin(T1+T2+T3+T4)))*((L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+  \
        T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+(L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*  \
        np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))**2)**(-1/2),
        (1/2)*(2*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(  \
        T1+T2+T3+T4))*((-1)*L3*np.sin(T1+T2+T3)+(-1)*L4*np.sin(T1+T2+T3+T4)  \
        )+2*(L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))*(L1*np.sin(T1)+L2*  \
        np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)))*((L1*np.cos(T1)  \
        +L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+(L1*  \
        np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))**2)  \
        **(-1/2),
        (1/2)*((-2)*L4*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+  \
        L4*np.cos(T1+T2+T3+T4))*np.sin(T1+T2+T3+T4)+2*L4*np.cos(T1+T2+T3+T4)*(  \
        L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))  \
        )*((L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+  \
        T3+T4))**2+(L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(  \
        T1+T2+T3+T4))**2)**(-1/2),
        (1/2)*(2*np.cos(T1)*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+  \
        L4*np.cos(T1+T2+T3+T4))+2*np.sin(T1)*(L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*  \
        np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)))*((L1*np.cos(T1)+L2*np.cos(T1+T2)  \
        +L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+(L1*np.sin(T1)+L2*np.sin(  \
        T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))**2)**(-1/2),
        (1/2)*(2*np.cos(T1+T2)*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+  \
        T3)+L4*np.cos(T1+T2+T3+T4))+2*np.sin(T1+T2)*(L1*np.sin(T1)+L2*np.sin(T1+  \
        T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)))*((L1*np.cos(T1)+L2*  \
        np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+(L1*np.sin(T1)  \
        +L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))**2)**(  \
        -1/2),
        (1/2)*(2*np.cos(T1+T2+T3)*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+  \
        T2+T3)+L4*np.cos(T1+T2+T3+T4))+2*np.sin(T1+T2+T3)*(L1*np.sin(T1)+L2*  \
        np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)))*((L1*np.cos(T1)  \
        +L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+(L1*  \
        np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4))**2)  \
        **(-1/2),
        (1/2)*(2*np.cos(T1+T2+T3+T4)*(L1*np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(  \
        T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))+2*np.sin(T1+T2+T3+T4)*(L1*np.sin(T1)+  \
        L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)))*((L1*  \
        np.cos(T1)+L2*np.cos(T1+T2)+L3*np.cos(T1+T2+T3)+L4*np.cos(T1+T2+T3+T4))**2+  \
        (L1*np.sin(T1)+L2*np.sin(T1+T2)+L3*np.sin(T1+T2+T3)+L4*np.sin(T1+T2+T3+T4)  \
        )**2)**(-1/2)])
    scaling = np.array([(b-a),(b-a),(b-a),(b-a),1,1,1,1])*0.5
    df = out*scaling
    return [f,df]
    