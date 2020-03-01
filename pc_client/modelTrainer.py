
import numpy as np
import cv2

win_shape = (512,512,3)
img = np.zeros(win_shape)

img = img + 255

x_len = 5
x_center = win_shape[0] // 2
x_coords = [
        (x_center - x_len, x_center - x_len), 
        (x_center + x_len, x_center + x_len), 
        (x_center + x_len, x_center - x_len),
        (x_center - x_len, x_center + x_len)]

img = cv2.line(img, x_coords[0], x_coords[1], (0,0,0), 2, lineType=cv2.LINE_AA)
img = cv2.line(img, x_coords[2], x_coords[3], (0,0,0), 2, lineType=cv2.LINE_AA)

cv2.imshow('modelTrain', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
