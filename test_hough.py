import cv2
import numpy as np
from math import cos, sin, sqrt
from ultils import *
from defaults import *

mask = []

img = cv2.imread(IMAGE_DIR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edgs = sobel_threshold(gray)
h, w = edgs.shape

h_array = 180 
w_array = int(sqrt(h**2 + w**2)) + 1
array = np.zeros((h_array, w_array), dtype=int)

print(array.shape)
threshold = 200

max = 0
for x in range(0, h):
    for y in range (0, w):
        for theta in range(1, h_array):
            p=x*cos(theta) + y*sin(theta)
            array[theta][int(p)] += 1
            
                        

        

