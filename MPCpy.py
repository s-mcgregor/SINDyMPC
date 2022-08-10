#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:55:01 2022

Description: Function library for MPC
             The optimal change in input (U) is defined by:
             delta_U = inv(Phi'Phi + R_bar)*Phi'(Rs_bar*r[k]-F*x[k])
             where:
                 inv(Phi'Phi + R_bar) is known as the Hessian Matrix
                 x[k] is the state information
                 r[k] is the set point (target)
             
             gatekeeper function is to enforce a consistent step time 
             
@author: Scott McGregor
"""
import numpy as np
import time

def mpcgain(Ap,Bp,Cp,Nc,Np):
    # Set up MPC design matrices
    m1 = Cp.shape[0]
    n1 = Cp.shape[1]
    n_in = Bp.shape[1]
    
    # Set A_e
    A_e = np.eye((n1+m1),(n1+m1))
    for x in range(0,n1):
        for y in range(0,n1):
            A_e[x,y] = Ap[x,y]

    CpAp = np.matmul(Cp,Ap)
    w = 0
    for x in range(n1, n1+m1):
        for y in range(0, n1):
            A_e[x,y] = CpAp[w,y]
        w = w+1
    
    # Set B_e
    B_e = np.zeros((n1+m1,n_in))
    for r in range(0,n1):
        for c in range(0,Bp.shape[1]):
            B_e[r,c] = Bp[r,c]
            
    CpBp = np.matmul(Cp,Bp)
    w = 0
    for x in range(n1, n1+m1):
        for y in range(0,Bp.shape[1]):
            B_e[x,y] = CpBp[w,y]
        w = w+1
    
    # Set C_e
    C_e = np.zeros((m1,n1+m1))
    C_eye = np.eye(m1,m1)
    w = 0
    for r in range(0,C_e.shape[0]):
        for c in range(n1,n1+m1):
            C_e[r,c] = C_eye[r,w]
        w = w+1
    
    ##
    n = n1+m1
    h = C_e
    F = np.matmul(C_e,A_e)
    hs = np.zeros((1,h.shape[1]))
    Fs = np.zeros((1,F.shape[1]))
    
    for kk in range(1,Np):
        h=np.vstack([h,hs]) # appends new row of zeros onto h for modification
        F=np.vstack([F,Fs]) # appends new row of zeros onto F for modification
        h[kk,:] = np.matmul(h[kk-1,:],A_e)
        F[kk,:] = np.matmul(F[kk-1,:],A_e)
        
    v = np.matmul(h,B_e)
    Phi = np.zeros([Np,Nc]) # declare the dimension of phi
    for r in range(0,Np): # declare first column
        Phi[r,0] = v[r,0]
        
    # Toeplitz Matrix
    for c in range(1,Nc): # resolve subsequent columns of phi
        zr = c
        for i in range(zr):
            Phi[i,c] = 0
        r=0
        for i in range(zr,Np):
            Phi[i,c] = v[r,0]
            r=r+1
    
    rw = 0.5 # tuning parameter
    Phi_T = Phi.T
    Rs_bar = np.ones((Np,1))
    R_bar = rw*np.eye(Nc,Nc)
    Phi_Phi = np.matmul(Phi_T,Phi)
    Phi_F = np.matmul(Phi_T,F)
    Phi_R = np.matmul(Phi_T, Rs_bar)
    inHessian = Phi_Phi + R_bar
    Hessian = np.linalg.inv(inHessian) 
      
    return A_e, B_e, C_e, Hessian, Phi_F, Phi_R


def gatekeeper(current_time, delay_length):
    ## Function to enforce a constant delay between the start of a loop and
    ## its completion. The idea is to not propagate delays due to calculation time.
    ## This function should be the last part of a calculation loop.
    # Inputs: current_time: start of function
    #         delay_length: seconds until next loop can begin
    target = current_time + delay_length
    while(time.time() < target):
        continue
    
    
'''
#################
### Unit Test ###
#################

## Unit test built to compare original MATLAB output to new code    
## Test Matrices
Ap=np.array([[1,2,3],[4,5,6],[7,8,9]])
Bp =np.array([[11],[12],[13]])
Cp=np.array([[1,2,3]])

Np = 3
Nc = 2

A_e, B_e, C_e, Hessian, Phi_F, Phi_R = mpcgain(Ap,Bp,Cp,Nc,Np)

xf = np.zeros(Ap.shape[0] + 1)[np.newaxis]
xf = xf.T
r = 5 # target value

# Step 1
side1= Phi_R*r
# Step 2
side2 = np.matmul(Phi_F,xf)
# Step 3
side1 = np.matmul(Hessian,side1)
# Step 4
side2 = np.matmul(Hessian,side2)
# Step 5
du = side1 - side2
'''
