from sre_constants import SUCCESS
import cv2
import numpy as np
from ultils import *
from defaults import *

img = cv2.imread(IMAGE_DIR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edgs = cv2.Canny(gray, 350, 500)
# show_image(edgs)
Hough_tranform(img, edgs, THRESHOLD)

# # show_image(img)
# save_image(img)
run_video(VIDEO_DIR)
# # # save_video(VIDEO_DIR)

    

