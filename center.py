import numpy as np
import argparse
import cv2
import serial
import time
from serialdata import *


def grayscale(img): 
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): 
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): 
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    x, y, gradient =  draw_lines(line_img, lines)
    #print(x, y, gradient)
    return line_img, x, y, gradient


def weighted_img(img, initial_img, a=1, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)
 
 
def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
   
    x=-1
    y=-1
    gradient=0
    maxvalue = 0
    point=[0,0,0,0]
    if lines is None:
        return x, y, gradient
   
    for line in lines:
        for x1,y1,x2,y2 in line:
                           
            if y2 < 120 and y1 < 120 :
                continue
            
             
            if maxvalue < max(y1, y2):
                maxvalue = max(y1, y2)
            
                gradient = (y2-y1)/(x2-x1+0.00001)
                
                x = max(x1, x2)
                point[0] = x1
                point[1] = y1
                point[2] = x2
                point[3] = y2
       
    if point[0] == 0 and point[1] == 0:
         for line in lines:
            for x1,y1,x2,y2 in line:
                               
                
                 
                if maxvalue < max(y1, y2):
                    maxvalue = max(y1, y2)
                
                    gradient = (y2-y1)/(x2-x1+0.00001)
                    
                    x = max(x1, x2)
                    point[0] = x1
                    point[1] = y1
                    point[2] = x2
                    point[3] = y2
            
        
        
    cv2.line(img, (point[0], point[1]), (point[2], point[3]), color, thickness)
            
    return x, y, gradient
    
def loop(serial_port) :
    
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 1  #PI CAMERA: 320 x 240 = MAX 90


    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)  
    

    TX_data_py2(serial_port, 29)
	
    while True:
        wait_receiving_exit()
        _,frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([50, 255, 255])
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        image_result = cv2.bitwise_and(frame, frame,mask = mask)
        cv2.imshow("a", image_result)
        cv2.waitKey(1)
        gray_img = grayscale(image_result)
        blur_img = gaussian_blur(gray_img, 3)
        canny_img = canny(blur_img, 20, 30)
        
        
        hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
        result = weighted_img(hough_img, frame)
        
        #print(gradient)
        
        
        cv2.imshow("img", result)
        cv2.waitKey(1)
        
 
        if  x == -1: break
        if abs(gradient) < 1: break 
        elif gradient>0 and gradient< 2.5:
            TX_data_py2(serial_port, 4)
            time.sleep(1)
            continue
        
        elif gradient<0 and gradient>-2.5:
            TX_data_py2(serial_port, 6) 
            time.sleep(1) 
            continue
        
        
        if  x > 180:
            TX_data_py2(serial_port, 20)
            
        elif x>10 and x < 140:
            TX_data_py2(serial_port, 15)
             
        elif x>=140 and x<=180:
            break 
            
        
            
        time.sleep(1) 
        

    cap.release()
    cv2.destroyAllWindows()
    
    time.sleep(1)
    exit(1)
    
if __name__ == '__main__':

    BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200

       
    serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
    serial_port.flush() # serial cls
    
    
    serial_t = Thread(target=Receiving, args=(serial_port,))
    serial_t.daemon = True
    
    
    serial_d = Thread(target=loop, args=(serial_port,))
    serial_d.daemon = True
    
    print("start")
    serial_t.start()
    serial_d.start()
    
   
    serial_d.join()
    print("end")
