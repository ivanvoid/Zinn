#!/usr/bin/python

import matplotlib.animation as anime
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import matplotlib
import serial

matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix'
    })


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def smooth_data(filt_order=3, cutoff_freq=0.1):
    global data
    B, A = signal.butter(filt_order, cutoff_freq, output='ba')
    data = signal.filtfilt(B, A, data)
    

def data_update():
    global data

    val = int(ser.readline().decode('ascii'))
    data = np.append(data, val)
    
    f.write(str(data[-1])+',')
   
    if len(data) > 100:
        data = data[1:]
        #smooth_data()

def animate(i):
    data_update()
    
    global data
    ax1.clear()
    ax1.plot(np.arange(data.size), data)


# MAIN

ser = serial.Serial('/dev/ttyUSB1', 9600, timeout=100)

data = np.array([])
f = open('signal_2_log', 'a+')

ani = anime.FuncAnimation(fig, animate, interval=1)
plt.show()

f.write('\n')
f.close()
print('End')    
ser.close()
