"""
Created on Sat Jul 16 23:03:37 2022

Description: Test to ensure proper working of SINDy MPC control algorithm
             This program takes an initial set of test data of a driven system.
             This information was human driven, with state information and input
             information recorded and organized into columns. The true discrete 
             time dynamics are written at Ad,Bd,Cd,Dd, however these are manually
             recorded to allow for system simulation.
             
             The control algorithm works by concurrently identifying the underlying
             system dynamics, while using these assumed model dynamics to generate
             an optimal control via a model predictive controller. Once the 
             dynamics are stabilized, the MPC will be able to give clean return values.
             
             Credit to Steve Brunton and the University of Washington dynamics lab
             for their work in developing the system identification SINDy technique,
             in which the system identification is based.

@author: Scott McGregor
"""

import numpy as np
import matplotlib.pyplot as plt
from SINDy import *
from MPCpy import *

## Set the "true" discrete time state-space matrices
#  These are considered unknown by the controller at initial time
Ad = np.array([[1.0, 0.0242, 0.2998, 0.0294],[0.0, -0.003, 0.6319, 0.1521],[0.0, -0.0056, 1.0035, 0.2840],[0.0, -0.0061, 0.0039, 0.2838]])
Bd = np.array([[0.0084],[0.0295],[0.0433],[0.1437]])
Cd = np.eye(4)
Dd = np.zeros((4,1))

## Load data for time series use
X1 = np.loadtxt("X1.txt",dtype="f")
X2 = np.loadtxt("X2.txt",dtype="f")

## Trim initial data
# This is just to deliver incomplete system dynamics on initial estimate
v = 850
X1 = X1[0:v,:]
X2 = X1[0:v,:]

## Set Parameters
# These parameters will hold constant throughout the analysis
polyorder = 1
usesine = 0
n = 5 # 4D system + control system 
s = X1.shape[1] - 1 # Number of states
dmax = 999 # max length of data to be stored

## Pool data (i.e. build library of nonlinear time series)
Theta = poolData(X1,n,polyorder,usesine)
m = Theta.shape[2-1] # Notes: size(a,n); a.shape[n-1]; m=size(Theta,2)

## Compute sparse regression
lambda_ = 0.005 # lambda is our sparsification knob
Xi = sparsifyDynamics(Theta, X2, lambda_,n)

## Set Plant Matrices
# These are the initial estimated plant matrices
 
# Set "A" and "B" matrices
Ats = Xi[1:n,0]
Bts = Xi[n,0]
for x in range(1,s):
    Ats = np.vstack([Ats,Xi[1:n,x]])
    Bts = np.vstack([Bts,Xi[n,x]])

# Set "C" matrix
Cts = np.zeros(n - 1) # Set correct size for the output matrix
Cts[0] = 1 # We are only interested in the first state in this example
Cts = np.expand_dims(Cts,axis=0) # Need to set matrix dimensions properly

# Set "D" matrix
Dts = 0.*np.matmul(Cts,Bts)

# Set prediction and control points
Np = 100
Nc = 5
Delta_t = 0.5

A_e, B_e, C_e, Hessian, Phi_F, Phi_R = mpcgain(Ats,Bts,Cts,Nc,Np)
[n,n_in] = B_e.shape

############
## Pass 1 ##
############

## Set these values for analysis (!!)
ic = 25 # initial condtion (!!)
xm = np.array([[ic],[0],[0],[0]]) # Set initial state conditions
xf = np.zeros((n,1))
N_sim = 250 # number of simulation points (!!)

r = 10 # target value for the MPC (!!)
u = 0 # u(k-1) = 0
y = ic
Old_Ats = Ats

u1=[]
y1=[]

## Simulate over number of simulation points
for kk in range(0,N_sim):
    
    # Calculate input delta
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
    deltau = du[0,0]
    u=u+deltau
    
    # Record input and meaurement info
    u1 = np.append(u1,u)
    y1 = np.append(y1,y)
    
    # Calculate system's response
    xm_old = xm
    xm = np.matmul(Ad,xm) + Bd*u
    y = np.matmul(Cts,xm)
    diffX = xm - xm_old
    xf = np.vstack([diffX,y])
    
    ## Reinitialize the system / SINDy based on new readings
    # Create temp files to be appended to data log
    duv = np.array([u])[np.newaxis]
    tX1 = np.hstack([xm_old.T,duv])
    one = np.array([1])[np.newaxis]
    tX2 = np.hstack([xm.T,one])
    # Append data logs
    X1 = np.vstack([X1,tX1])
    X2 = np.vstack([X2,tX2])
    # Check if data exceeds size
    # This is the rolling window of data to be processed. 
    if X1.shape[0] > dmax:
        # remove first row of X1 and X2
        X1 = np.delete(X1,0,0)
        X2 = np.delete(X2,0,0)
    
    ## Rerun data pool (rebuild Theta library)
    Theta = poolData(X1,n,polyorder,usesine)
    Xi = sparsifyDynamics(Theta, X2, lambda_,n)
    
    ## Update Estimated Plant Matrices
    # Copied from earlier code
    
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
    
    ## Update information for the MPC
    A_e, B_e, C_e, Hessian, Phi_F, Phi_R = mpcgain(Ats,Bts,Cts,Nc,Np)
    

    
# Generate output plots    
kk = []
for x in range(0,N_sim):
    kk = np.append(kk,x)
    
plt.plot(kk,y1)
plt.xlabel('Sampling Instant')
plt.ylabel('Output')
plt.title("Pass 1 - Output")
plt.show()


##########
# Pass 2 #
##########   

# Code copied from Pass 1, with new conditions
# This pass is to demonstrate how the internal dynamics have been learned.
# They can then be utilized for future MPC input selection with different values.

## Set these values for analysis (!!)
ic = 10 # initial condtion (!!)
xm = np.array([[ic],[0],[0],[0]]) # Sets initial state conditions
xf = np.zeros((n,1))
N_sim = 250 # number of simulation points (!!)

r = 30 # target value for the MPC (!!)
u = 0 # u(k-1) = 0
y = ic
Old_Ats = Ats

u1=[]
y1=[]

## Simulate over number of simulation points
for kk in range(0,N_sim):
    
    # Calculate input delta
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
    deltau = du[0,0]
    u=u+deltau
    
    # Record input and meaurement info
    u1 = np.append(u1,u)
    y1 = np.append(y1,y)
    
    # Calculate system's response
    xm_old = xm
    xm = np.matmul(Ad,xm) + Bd*u
    y = np.matmul(Cts,xm)
    diffX = xm - xm_old
    xf = np.vstack([diffX,y])
    
    ## Reinitialize the system / SINDy based on new readings
    # Create temp files to be appended to data log
    duv = np.array([u])[np.newaxis]
    tX1 = np.hstack([xm_old.T,duv])
    one = np.array([1])[np.newaxis]
    tX2 = np.hstack([xm.T,one])
    # Append data logs
    X1 = np.vstack([X1,tX1])
    X2 = np.vstack([X2,tX2])
    # Check if data exceeds size
    # This is the rolling window of data to be processed. 
    if X1.shape[0] > dmax:
        # remove first row of X1 and X2
        X1 = np.delete(X1,0,0)
        X2 = np.delete(X2,0,0)
    
    ## Rerun data pool (rebuild Theta library)
    Theta = poolData(X1,n,polyorder,usesine)
    Xi = sparsifyDynamics(Theta, X2, lambda_,n)
    
    ## Update Estimated Plant Matrices
    # Copied from earlier code
    
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
    
    ## Update information for the MPC
    A_e, B_e, C_e, Hessian, Phi_F, Phi_R = mpcgain(Ats,Bts,Cts,Nc,Np)
    


    
# Generate output plots    
kk = []
for x in range(0,N_sim):
    kk = np.append(kk,x)
    
plt.plot(kk,y1)
plt.xlabel('Sampling Instant')
plt.ylabel('Output')
plt.title("Pass 2 - Output")
plt.show()
    
    