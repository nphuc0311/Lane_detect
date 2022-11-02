import numpy as np
import cv2
import math

sum_x = 0
sum_y = 0
binary_output = []
mag_thresh = (30,255)
ang_thresh = (.8, 1.4)

gx = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])

gy = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])

img = cv2.imread('./test_images/straight_lines2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x_sobel = []
y_sobel = []

h, w = gray.shape
for i in range(0, h):
    for j in range(0, w):
        mask = gray[i:i+3, j:j+3]
        h_mask, w_mask = mask.shape
        for k in range(0, h_mask):
            for l in range(0, w_mask):
                t_x = gx[k][l]*mask[k][l]
                sum_x = sum_x + t_x
                t_y = gy[k][l]*mask[k][l]
                sum_y = sum_y + t_y
        x_sobel.append(sum_x)
        y_sobel.append(sum_y)
x_sobel = np.array(x_sobel)
y_sobel = np.array(y_sobel)
grad_ang = np.arctan2(y_sobel, x_sobel)

grad_mag = np.sqrt(x_sobel**2 + y_sobel**2)
# grad_mag = grad_mag.reshape(grad_mag,(720,1280))
print(grad_mag)
# scale_factor = np.max(grad_mag)/255
# grad_mag = (grad_mag/scale_factor).astype(np.uint8)

# binary_output[
#         (grad_mag > mag_thresh[0]) & 
#         (grad_mag < mag_thresh[1]) &
#         (grad_ang > ang_thresh[0]) & 
#         (grad_ang < ang_thresh[1])
#     ] = 255
            
# cv2.imshow('image', binary_output)
# cv2.waitKey()
# cv2.destroyAllWindows()


