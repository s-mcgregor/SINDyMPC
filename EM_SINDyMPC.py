#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:05:44 2022

Description: Hardware implementation of the SINDyMPC control. (EM_SINDyMPC)
             The controlled implement is an electric motor with unknown parameters.
             The goal is to implement the control scheme and achieve performance
             of interest.

             State Information:
             Angular velocity
             Motor current
             Applied voltage (control variable)

@author: Scott McGregor
"""

import RPi.GPIO as GPIO
from ina219 import INA219
import Encoder
from math import pi
import time
import numpy as np
import matplotlib.pyplot as plt
from SINDy import *
from MPCpy import *

## GPIO pin usage ##
#GPIO 17 Motor pin A
#GPIO 27 Motor pin B
#GPIO 22 Moter Enable Pin
#GPIO 06 Encoder A
#GPIO 13 Encoder B
#GPIO 02 SDA (ina 219)
#GPIO 03 SCL (ina 219)


## Load data for time series use
# These are the original created files from Basic_Motor_Control.py
X1 = np.load('bat_X0.npy')
X2 = np.load('bat_X1.npy')


## Set up motor and sensors
# Battery voltage
battery = 9 # volts

# Initial inforation for the ina219 chip
ina = INA219(shunt_ohms=0.1, max_expected_amps = 0.6, address=0x40)

ina.configure(voltage_range=ina.RANGE_16V,
              gain=ina.GAIN_AUTO,
              bus_adc=ina.ADC_128SAMP,
              shunt_adc=ina.ADC_128SAMP)

# Initialize Matplot tracking variables
t = [] # initialize x-plot, which will be time
m_speed = [] # initialize y-plot #1, which will be motor speed
m_current = [] # # initialize y-plot #2, which will be motor current
input_voltage = [] # initialize array of input voltage signals
r_list = [] # initialize array of target values

# Set source values for encoder reading 
# This information will be processed into motor speed
GPIO.setmode(GPIO.BCM) # Use BCM for GPIO pin selection
GPIO.setup(6,GPIO.IN)
GPIO.setup(13,GPIO.IN)
enc = Encoder.Encoder(6,13)
ppr =  2125 #2125 # encoder pulses per rotation (after gearing), found experimentally

# Initialize motor
MotorA = 17
MotorB = 27
MotorE = 22
 
GPIO.setup(MotorA,GPIO.OUT)
GPIO.setup(MotorB,GPIO.OUT)
GPIO.setup(MotorE,GPIO.OUT)

#Setup PWM on motor speed
fq = 500
pwm_motor = GPIO.PWM(MotorE,fq)
duty = 100

fq_counter = 0 
fq_control = 40 # adjustable value
fq_cnt_tar = 500/fq_control

## Set SINDyMPC Parameters
# These parameters will hold constant throughout the analysis
polyorder = 1
usesine = 0
n = 3 # 4D system + control system 
s = X1.shape[1] - 1 # Number of states
dmax = 9999 # max length of data to be stored

## Pool data (i.e. build library of nonlinear time series)
Theta = poolData(X1,n,polyorder,usesine)
m = Theta.shape[2-1] # Notes: size(a,n); a.shape[n-1]; m=size(Theta,2)

## Compute sparse regression
lambda_ = 0.005 # lambda is our sparsification knob
Xi = sparsifyDynamics(Theta, X2, lambda_,n)

## Set Plant Matrices
# These are the initial estimated plant matrices
Ats, Bts, Cts, Dts = estimatePlants(Xi,n,s)

# Set prediction and control points
Np = 100
Nc = 10
Delta_t = 0.02 # 50Hz delta time

A_e, B_e, C_e, Hessian, Phi_F, Phi_R = mpcgain(Ats,Bts,Cts,Nc,Np)
[n,n_in] = B_e.shape

## Begin motor movement
pwm_motor.start(duty)
GPIO.output(MotorA,True)
GPIO.output(MotorB, False)
GPIO.output(MotorE,True)

## Set these values for analysis (!!)
ic = 8 # initial condtion (!!)
xm = np.array([[ic],[0]]) # Set initial state conditions
xf = np.zeros((n,1))

r = 4 # target value for the MPC (!!)
u = 0 # u(k-1) = 0
y = ic

# Set initial time information
vtime = 0
u = duty
time_prev = time.time()
enc_prev = enc.read()
time.sleep(1)

while vtime < 20:
    
    if vtime >= 10:
        r = 7
    
    #Measure motor speed/currents
    time_now = time.time()
    enc_now = enc.read()
    time_delta = time_now - time_prev
    enc_delta = enc_now - enc_prev
    w = 2*pi*(enc_delta / time_delta)*(1/ppr) ## angular vel; units:rad/s
    
    i = ina.current() # motor current (mA)
    app_voltage = (duty/100)*battery # applied voltage to motor
    
    if fq_counter == 10: #[50Hz] (500/fq_control):
        
        # Calculate system's response
        xm_old = xm               # previous state information
        xm = np.array([[w],[i]])  # record current state info
        y = w                     # tracked variable
        diffX = xm - xm_old       # difference in states
        xf = np.vstack([diffX,y]) 
        
        ## Reinitialize the system / SINDy based on new readings
        # Create temp files to be appended to data log
        duv = np.array([duty])[np.newaxis] # add new input 'u'
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
        Ats, Bts, Cts, Dts = estimatePlants(Xi,n,s)
        
        ## Update information for the MPC
        A_e, B_e, C_e, Hessian, Phi_F, Phi_R = mpcgain(Ats,Bts,Cts,Nc,Np)
        
        ## Calculate input delta
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
        
        ## Update duty cycle
        duty = u
        # set minimum input value
        try:
            pwm_motor.ChangeDutyCycle(duty)
        except:
            pwm_motor.ChangeDutyCycle(20)
            u = 20
            
        #record data
        t.append(vtime)
        m_speed.append(w)
        m_current.append(i)
        input_voltage.append(app_voltage)
        r_list.append(r)
        fq_counter = 0 # reset freq counter
    
    # reset time and encoder variables
    time_prev = time_now
    enc_prev = enc_now
    
    #advance time
    fq_counter += 1
    vtime += (1/fq)
    gatekeeper(time_now, (1/fq))


## Clean Up 
GPIO.output(MotorE,False) 
GPIO.cleanup() #Cleanup GPIO output

## create graphical output
fig , axs = plt.subplots(2,1)
axs[0].plot(t,m_speed)
axs[0].plot(t,r_list)
axs[0].set_ylabel('Ang Vel (radps)')
axs[1].plot(t,m_current)
axs[1].set_ylabel('Current (mA)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()