#!/usr/bin/env python3
# -*-  coding: utf-8 -*-

import numpy as np
import serial
import cv2

def render(img):
    # User cross target
    x_len = 15
    x_center = img.shape[0] // 2
    x_coords = [
        (x_center - x_len, x_center - x_len), 
        (x_center + x_len, x_center + x_len), 
        (x_center + x_len, x_center - x_len),
        (x_center - x_len, x_center + x_len)]

    # AI circle prediction
    ai_coords = (256,256)
    r = 10


    # Render objects
    img = cv2.circle(img, ai_coords, r, (0,255,0), -1)

    img = cv2.line(img, x_coords[0], x_coords[1], (0,0,0), 
        2, lineType=cv2.LINE_AA)
    img = cv2.line(img, x_coords[2], x_coords[3], (0,0,0), 
        2, lineType=cv2.LINE_AA)
    img = cv2.circle(img, (x_center,x_center), 5, (255,255,255), -1)
    img = cv2.circle(img, (x_center,x_center), 2, (0,0,0), 
            -1, lineType=cv2.LINE_AA)
    
    return img


def read_signal(serialport, data, max_len = 100):
    value = int(serialport.readline().decode('ascii'))
    data = np.append(data, value)

    if data.size > max_len:
        data = data[1:]
    
    return data

def main():
    win_shape = (512,512,3)
    img = np.zeros(win_shape) + 255

    path = '/dev/ttyUSB1'
    freq = 9600
    tout = 100

    ser = serial.Serial(path, freq, timeout=tout)

    data = np.array([])

    while True:
        data = read_signal(ser, data)
        print(data[-1])
        
        img = render(img)
        cv2.imshow('modelTrain', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

main()

