#!/usr/bin/env python3
# -*-  coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import numpy as np
import serial
import cv2

# Ignore warnings
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class User():
    '''
        This class for target manipulation on PC
    '''
    def __init__(self, width, height):
        self.length = 10
        self.x = width // 2 
        self.y = height // 2
        self.cross_coords = [
            (self.x - self.length, self.y - self.length),
            (self.x + self.length, self.y + self.length),
            (self.x - self.length, self.y + self.length),
            (self.x + self.length, self.y - self.length)]

    def _recompute_cross(self):
        self.cross_coords = [
            (self.x - self.length, self.y - self.length),
            (self.x + self.length, self.y + self.length),
            (self.x - self.length, self.y + self.length),
            (self.x + self.length, self.y - self.length)]

    def get_coordinats(self):
        return np.array([self.x, self.y])

    def set_x(self, x):
        self.x = x
        self._recompute_cross()

    def set_y(self, y):
        self.y = y
        self._recompute_cross()


class Model():
    def __init__(self):
        self.x = 0
        self.y = 0
        #self.lr = LinearRegression()
        self.lr = MLPRegressor(
                hidden_layer_sizes=(50,50),
                learning_rate_init=0.001)

    def fit(self, X, y):
        self.lr.fit(X, y)

    def predict(self, X):
        y_pred = self.lr.predict(X)
#        print('Predict (evaluate->retrain)')
        return y_pred

def render(img, user, pred_xy):
    # Model circle radius 
    r = 10

    # Render objects
    img = cv2.circle(img, pred_xy, r, (0,255,0), -1)

    img = cv2.line(img, user.cross_coords[0], user.cross_coords[1], 
        (0,0,0), 2, lineType=cv2.LINE_AA)
    img = cv2.line(img, user.cross_coords[2], user.cross_coords[3], 
        (0,0,0), 2, lineType=cv2.LINE_AA)
    img = cv2.circle(img, (user.x,user.y), 5, (255,255,255), -1)
    img = cv2.circle(img, (user.x,user.y), 2, (0,0,0), 
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

def log(user_xy, y_pred, ser_signal):
    print('U:({u[0]:3d}, {u[1]:3d}) | Pred:({y[0]:3d},{y[1]:3d})'
            '| Signal:{s}'.format(
            u = user_xy, # user 
            y = y_pred,     # y predicted
            s = ser_signal))  # signal


def main():
    MODE = 'train'
    # Create white backgroung img
    win_shape = (512,512,3)
    img = np.zeros(win_shape) + 255

    # Setup serial 
    path = '/dev/ttyUSB1'
    freq = 9600
    tout = 100
    ser = serial.Serial(path, freq, timeout=tout)

    # Creating User
    user = User(img.shape[1], img.shape[0])

    # Creating ai model
    model = Model()

    # Here we store data from serial input
    data = np.array([])
    u_data = [user.get_coordinats()]

    i = 0 # Waiting for more data to be collected

    while True:
        data = read_signal(ser, data)
        u_data = get_user_data(user, u_data)
     
        if MODE == 'train':
            if i >= 100:
                model.fit(data.reshape(-1,1), u_data)
                y_pred = model.predict(data[-1].reshape(-1,1))
                y_pred = tuple(y_pred[0].astype(int))
            else:
                y_pred = (0,0)
                i += 1
        
            # Logs
            log(u_data[-1], y_pred, data[-1])
        
        elif MODE == 'test':
            y_pred = model.predict(data[-1].reshape(-1,1))
            y_pred = tuple(y_pred[0].astype(int))
    
            print('TEST: ', y_pred)

            user.set_x(-12)
            user.set_y(-12)

        # Window settings
        img = render(img, user, y_pred)
        win_name = 'modelTrainer'
        cv2.imshow(win_name, img)
        cv2.moveWindow(win_name, 2,2)
        
        # Clear img
        img = np.zeros(win_shape) + 255

        # Keys listener
        k = cv2.waitKey(10)
        speed = 10

        if k == 27 or k == 113: # Esc or q
            break
        elif k == 119: # w
            user.set_y(user.get_coordinats()[1] - speed)
        elif k == 115: # s
            user.set_y(user.get_coordinats()[1] + speed)
        elif k == 97:  # a
            user.set_x(user.get_coordinats()[0] - speed)
        elif k == 100: # d
            user.set_x(user.get_coordinats()[0] + speed)
        elif k == 109: # m. Switch mode train/test
            if MODE == 'train': 
                MODE = 'test'
            else:
                MODE = 'train'
                user.set_x(50)
                user.set_y(50)
        #else:
        #    print(k)
        
    cv2.destroyAllWindows()

main()

