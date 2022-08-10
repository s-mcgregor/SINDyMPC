"""
Created on Wed Jul  6 21:42:36 2022
Description: Function library for Sparse Identification (SINDy)
             Code based on MATLAB functions written by Prof Steve Brunton.
             
             The function works by creating a linear or nonlinear library of
             potential terms for the state matrix. This is listed as Theta,
             and is built in the poolData function. The sparsifyDynamics 
             function performs a regression to solve the equation
             X' = Theta(X)*Xi, where Xi is the estimated system dynamics.

@author: Scott McGregor
"""

import numpy as np
from scipy import sparse
from sklearn.linear_model import Lasso

def poolData(yin,nVars,polyorder,usesine):
    ## Builds the Theta matrix that is the library of state terms,
     # which will be used to regress the system dynamics.
    
    n = yin.shape[0]
    
    # Poly Order 0
    yout = np.ones((n,1))
    
    # Poly Order 1
    # Output: [1 X Y]
    for x in range(0,nVars):
        a = np.array([yin[:,x]]).T
        yout = np.append(yout,a,axis=1)
    
    # Poly Order 2
    # Output: [1 X Y X^2 X*Y Y^2]
    if polyorder >= 2:
        for i in range(0,nVars):
            for j in range(i,nVars):
                q = np.array([yin[:,i]]).T
                w = np.array([yin[:,j]]).T
                e = q*w
                yout = np.append(yout,e,axis=1)
                
    # Poly Order 3
    if polyorder >= 3:
        for i in range(0,nVars):
            for j in range(i,nVars):
                for k in range(j,nVars):
                    q = np.array([yin[:,i]]).T
                    w = np.array([yin[:,j]]).T
                    e = np.array([yin[:,k]]).T
                    r = q*w*e
                    yout = np.append(yout,r,axis=1)
    
    # Poly Order 4
    if polyorder >= 4:
        for i in range(0,nVars):
            for j in range(i,nVars):
                for k in range(j,nVars):
                    for l in range(k,nVars):
                        q = np.array([yin[:,i]]).T
                        w = np.array([yin[:,j]]).T
                        e = np.array([yin[:,k]]).T
                        r = np.array([yin[:,l]]).T
                        t = q*w*e*r
                        yout = np.append(yout,t,axis=1)
                        
    
    # Poly Order 5
    if polyorder >= 5:
        for i in range(0,nVars):
            for j in range(i,nVars):
                for k in range(j,nVars):
                    for l in range(k,nVars):
                        for m in range(l,nVars):
                            q = np.array([yin[:,i]]).T
                            w = np.array([yin[:,j]]).T
                            e = np.array([yin[:,k]]).T
                            r = np.array([yin[:,l]]).T
                            t = np.array([yin[:,m]]).T
                            y = q*w*e*r*t
                            yout = np.append(yout,y,axis=1)
    
    if usesine == 1:
        for k in range(1,11):
            s = np.sin(k*yin)
            c = np.cos(k*yin)
            yout = np.append(yout,s,axis=1)
            yout = np.append(yout,c,axis=1)
                        
    return yout




def sparsifyDynamics(Theta, dXdt, lambda_, n):
    # Different regression algorithms are listed below
    # Each have their own advantages
    
    # Compute Regression: Least Squares Regression
    Xi =  np.linalg.lstsq(Theta,dXdt,rcond=None)[0]
    
    # Compute Sparse Regression: LASSO sparse regression
    # reg = Lasso(alpha=lambda_)
    # reg.fit(Theta,dXdt)
    # Xi = reg.coef_
    # Xi = Xi.T
    
    
    ### Additional Notes ###
    # sparsifyDynamics Matrix Inversion Test/Notes #

    # Compute Sparse Regression: psuedoinverse
    # Xi = np.linalg.pinv(Theta).dot(dXdt) # old solver method, resulted in failures
    # sTheta = sparse.csr_matrix(Theta) # Solver attempt #2: Only worked for square matrices
    # sdXdt = sparse.csr_matrix(dXdt) # Solver attempt #2: Only worked for square matrices
    # Xi = spsolve(sTheta,sdXdt) # Solver attempt #2: Only worked for square matrices
        
    return Xi



def estimatePlants(Xi,n,s):
    ## Update Estimated Plant Matrices
    # Inputs: Xi: Regressed dynamics matrix
    #          n: System size + control
    #          s: Number of states
    
    # Set "A" and "B" matrices
    Ats = Xi[1:n,0]
    Bts = Xi[n,0]
    for x in range(1,s):
        Ats = np.vstack([Ats,Xi[1:n,x]])
        Bts = np.vstack([Bts,Xi[n,x]])
    
    # Set "C" matrix
    Cts = np.zeros(n - 1) 
    Cts[0] = 1 # We are only interested in the first state in this example
    Cts = np.expand_dims(Cts,axis=0)

    # Set "D" matrix
    Dts = 0.*np.matmul(Cts,Bts)
    
    return Ats, Bts, Cts, Dts


'''
#################
### Unit Test ###
#################

## Unit test built to compare original MATLAB output to new code        
## Test Data
polyorder = 1
usesine = 0
n = 2 # 2D system
yin = np.array([[1,2,3,4,5],[6,7,8,9,10]])
yin = yin.transpose()
X2 = np.array([[2,3,4,5,6],[7,8,9,10,11]])
X2 = X2.transpose()

## Pool data (Build library of nonlinear time series)
Theta = poolData(yin,n,polyorder,usesine)
m = Theta.shape[1]
# print(Theta)

## Compute sparse regression: sequential least squares
lambda_ = 0.005 # lamba is the sparsification knob
Xi =  np.linalg.lstsq(Theta,X2,rcond=None)[0]
print(Xi)
reg = Lasso(alpha=0.1)
reg.fit(Theta,X2)
reg.coef_
'''

