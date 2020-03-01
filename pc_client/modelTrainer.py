#!/usr/bin/env python3
# -*-  coding: utf-8 -*-

from sklearn.linear_model import LinearRegression

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
 
    def get_coordinats(self):
        return np.array([self.x, self.y])

class Model():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(X, y)
        #        print('Fit')

    def predict(self, X):
        y_pred = self.lr.predict(X)
#        print('Predict (evaluate->retrain)')
        return y_pred

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

def get_user_data(user, u_data, max_len=100):
    u_data = np.append(u_data,[user.get_coordinats()], axis=0)
    
    if u_data.shape[0] > max_len:
        u_data = u_data[1:]

    return u_data

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
    u_data = [user.get_coordinats()]

    i = 0
    while True:
        data = read_signal(ser, data)
        u_data = get_user_data(user, u_data)
     
#        if i >= 100:
#            model.fit(data.reshape(-1,1), user_coords)
#            y_pred = model.predict(data)
#        else:
#            i += 1
        
        print('Data shape',data.reshape(-1,1).shape)
        print('User data shape', u_data.shape)
        y_pred=(42,42)
        
        img = render(img, user, y_pred)
        cv2.imshow('modelTrain', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

main()

