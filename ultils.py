import cv2
import numpy as np
import math
import os
from defaults import *


    
def sobel_threshold(img, mag_thresh = (30,255),ang_thresh = (.8, 1.4)):
    
    # Apply x and y gradient with the OpenCV Sobel() function and take abs value
    x_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    y_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    
    # absolute value of direction of gradient
    grad_ang = np.arctan2(y_sobel, x_sobel)
    
    # magnitude of direction of gradient
    grad_mag = np.sqrt(y_sobel**2 + x_sobel**2)
    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag/scale_factor).astype(np.uint8) 
    
    
    
    # Create a copy of sobel
    # Sobel is 1 px smaller on all sides so img will not work here
    binary_output = np.zeros_like(x_sobel).astype(np.uint8)
    
    # Apply Gradient
    #binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1]) &
    #            (abs_grad_dir >= ang_thresh[0]) & (abs_grad_dir <= ang_thresh[1])] = 1
    binary_output[
        (grad_mag > mag_thresh[0]) & 
        (grad_mag < mag_thresh[1]) &
        (grad_ang > ang_thresh[0]) & 
        (grad_ang < ang_thresh[1])
    ] = 255

    return binary_output

def Hough_tranform(img, edgs, thres=230):
    lines = cv2.HoughLines(edgs, 1, np.pi/180, thres)
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def save_image(img):
    cv2.imwrite(os.path.join(SAVE_DIR, 'linesDetected.jpg'), img)

def run_video(dir):
    video = cv2.VideoCapture(dir)
    while True:
        ret, frame = video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edgs = cv2.Canny(gray, 250, 400)
            # edgs = sobel_threshold(gray)
            Hough_tranform(frame, edgs, THRESHOLD)

            cv2.imshow('Canny', edgs)
            cv2.imshow('Hough', frame)
            if (i == 0):
                save_image(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()

def save_video(dir):
    video = cv2.VideoCapture(dir)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(os.path.join(SAVE_DIR, 'output_video.avi'), 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, size)

    while True:
        ret, frame = video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edgs = sobel_threshold(gray)
            Hough_tranform(frame, edgs, THRESHOLD)
            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break
    video.release()
    result.release()
    cv2.destroyAllWindows()

