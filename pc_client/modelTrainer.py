
import numpy as np
import cv2

win_shape = (512,512,3)
img = np.zeros(win_shape)

img = img + 255

# User cross target
x_len = 15
x_center = win_shape[0] // 2
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
img = cv2.circle(img, (x_center,x_center), 2, (0,0,0), -1, lineType=cv2.LINE_AA)

cv2.imshow('modelTrain', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
