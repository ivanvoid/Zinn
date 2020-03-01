#!/usr/bin/env python3
# -*-  coding: utf-8 -*-

import numpy as np
import serial
import cv2

class User():
    def __init__(self, width, height):
        self.length = 10
        self.center = (width // 2, height // 2) 
        self.cross_coords = [
            (self.center[0] - self.length, self.center[1] - self.length),
            (self.center[0] + self.length, self.center[1] + self.length),
            (self.center[0] - self.length, self.center[1] + self.length),
            (self.center[0] + self.length, self.center[1] - self.length)]
        self.x = self.center[0]
        self.y = self.center[1]
 
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y

class Model():
    def __init__(self):
        self.x = 0
        self.y = 0

    def fit(self, data):
        print('Fit')

    def predict(self, data):
        print('Predict (evaluate->retrain)')
        return 42,42

def render(img, user, pred_xy):
    # AI circle prediction
    ai_coords = (256,256)
    r = 10

    # Render objects
    img = cv2.circle(img, ai_coords, r, (0,255,0), -1)

    img = cv2.line(img, user.cross_coords[0], user.cross_coords[1], 
        (0,0,0), 2, lineType=cv2.LINE_AA)
    img = cv2.line(img, user.cross_coords[2], user.cross_coords[3], 
        (0,0,0), 2, lineType=cv2.LINE_AA)
    img = cv2.circle(img, user.center, 5, (255,255,255), -1)
    img = cv2.circle(img, user.center, 2, (0,0,0), 
            -1, lineType=cv2.LINE_AA)
    
    return img


def read_signal(serialport, data, max_len = 100):
    value = int(serialport.readline().decode('ascii'))
    data = np.append(data, value)

    if data.size > max_len:
        data = data[1:]
    
    return data

def main():
    # Create white backgroung img
    win_shape = (512,512,3)
    img = np.zeros(win_shape) + 255

    # Setup serial 
    path = '/dev/ttyUSB1'
    freq = 9600
    tout = 100
    ser = serial.Serial(path, freq, timeout=tout)

    # Creating User
    user = User(img.shape[0], img.shape[0])

    # Creating ai model
    model = Model()

    # Here we store data from serial input
    data = np.array([])
    user_coords = np.array([])

    while True:
        data = read_signal(ser, data)
        print(data[-1])
        user_coords = np.append(user_coords, [user.get_x(), user.get_y()])
      
        model.fit(data)
        x, y = model.predict(data)

        img = render(img, user, (x,y))
        cv2.imshow('modelTrain', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

main()

