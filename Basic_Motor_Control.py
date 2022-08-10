#Project:     Basic Motor Control
#Description: Outline the basic function call to control and record information
#             from a motor using sensors of interest.

import RPi.GPIO as GPIO
from ina219 import INA219
import Encoder
from math import pi
import time
import numpy as np
import matplotlib.pyplot as plt

def gatekeeper(current_time, delay_length):
    ## Function to enforce a constant delay between the start of a loop and
    ## its completion. The idea is to not propagate delays due to calculation time.
    ## This function should be the last part of a calculation loop.
    # Inputs: current_time: start of function
    #         delay_length: seconds until next loop can begin
    # Function copied from MPCpy
    target = current_time + delay_length
    while(time.time() < target):
        continue

## GPIO pin usage ##
#GPIO 17 Motor pin A
#GPIO 27 Motor pin B
#GPIO 22 Moter Enable Pin
#GPIO 06 Encoder A
#GPIO 13 Encoder B
#GPIO 02 SDA (ina 219)
#GPIO 03 SCL (ina 219)

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
pwm_motor.start(duty)

# Begin motor movement
GPIO.output(MotorA,True)
GPIO.output(MotorB, False)
GPIO.output(MotorE,True)
print('Going forwards')

# Working on a frequency counter allows for 
fq_counter = 0 
fq_control = 50
fq_cnt_tar = 500/fq_control

vtime = 0
flag2 = 0
flag4 = 0
flag6 = 0
flag8 = 0
time_prev = time.time()
enc_prev = enc.read()
time.sleep(1)

while vtime < 10:
    
    # update cycle value to demo speed change
    # preplanned duty cycles to record values throughout operational range
    if vtime > 2:
        if flag2 == 0:
            print('Speed Adjustment -40%')
            duty = 60
            pwm_motor.ChangeDutyCycle(duty)
            flag2 = 1
    if vtime > 4:
        if flag4 == 0:
            print('Speed Adjustment -40%')
            duty = 20 # treat a duty cycle of 20% as the 'floor' (motor stall)
            pwm_motor.ChangeDutyCycle(duty)
            flag4 = 1
    if vtime > 6:
        if flag6 == 0:
            print('Speed Adjustment +20%')
            duty = 40
            pwm_motor.ChangeDutyCycle(duty)
            flag6 = 1
    if vtime > 8:
        if flag8 == 0:
            print('Speed Adjustment +40%')
            duty = 80
            pwm_motor.ChangeDutyCycle(duty)
            flag8 = 1
            
    #Measure motor speed/currents
    time_now = time.time()
    enc_now = enc.read()
    time_delta = time_now - time_prev
    enc_delta = enc_now - enc_prev
    w = 2*pi*(enc_delta / time_delta)*(1/ppr) ## angular vel; units:rad/s
    
    i = ina.current() # motor current (mA)
    app_voltage = (duty/100)*battery # applied voltage to motor
    
    if fq_counter == 10: #fq_cnt_tar: #(500/fq_control):
        #record data
        t.append(vtime)
        m_speed.append(w)
        m_current.append(i)
        input_voltage.append(duty) # recording data to be agnostic to the full battery voltage
        fq_counter = 0 # reset freq counter
    
    # reset time and encoder variables
    time_prev = time_now
    enc_prev = enc_now
    
    #advance time
    fq_counter += 1
    vtime += (1/fq)
    gatekeeper(time_now, (1/fq))

print('Stop')
GPIO.output(MotorE,False) 
GPIO.cleanup() #Cleanup GPIO output


## Build output matrix ##
# State1 [motor speed]
X0_w = np.array([m_speed[0:len(m_speed)-1]])
X0_w = X0_w.T
X1_w = np.array([m_speed[1:len(m_speed)]])
X1_w = X1_w.T
# State2 [motor current]
X0_i = np.array([m_current[0:len(m_current)-1]])
X0_i = X0_i.T
X1_i = np.array([m_current[1:len(m_current)]])
X1_i = X1_i.T

# Input(Control)
U0 = np.array([input_voltage[0:len(input_voltage)-1]])
U0 = U0.T
U1 = np.array([input_voltage[1:len(input_voltage)]])
U1 = U1.T


# Combine matrix 
X0 = np.hstack([X0_w, X0_i, U0])
X1 = np.hstack([X1_w, X1_i, U1])
# Save data
np.save("bat_X0.npy",X0)
np.save("bat_X1.npy",X1)

print(X0.shape)

## create graphical output
fig , axs = plt.subplots(2,1)
axs[0].plot(t,m_speed)
axs[0].set_ylabel('Ang Vel (radps)')
axs[1].plot(t,m_current)
axs[1].set_ylabel('Current (mA)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()


'''
arr = np.array([[1,2,3,4,5]])
np.save("sample.npy",arr)
np.load('sample.npy')

state information
angular velocity
motor current
applied voltage (control variable)
'''                