import platform
import numpy as np
import argparse
import cv2
import serial
import time
from serialdata import *

def loop(serial_port):
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 10  #PI CAMERA: 320 x 240 = MAX 90

    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)  
    
    left_count = 0
    right_count = 0
    
    time.sleep(1)
    TX_data_py2(serial_port, 32)
    time.sleep(1)
    TX_data_py2(serial_port, 43)
    TX_data_py2(serial_port, 54)
    
    Flag = False
    
    while True:
        wait_receiving_exit()
        _,frame = cap.read()

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 0, 0])
        upper_red = np.array([180, 236, 52])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel)

        contours,_ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)
          
            
            if area > 2000 and area < 320*180:
                
                if Flag is False:
                    left = list(tuple(cnt[cnt[:, :, 0].argmin()][0]))
                    right = list(tuple(cnt[cnt[:, :, 0].argmax()][0]))
                    print("--------------")
                    print(left[0])
                    print(right[0])
                    
                    if int(left[0]) < 2 : 
                        TX_data_py2(serial_port, 62)
                        time.sleep(1)
                        Flag = True
                        
                    if int(right[0])> 318 : 
                        TX_data_py2(serial_port, 63)
                        time.sleep(1)
                        Flag = True
                        
                points = []
               
                if len(approx)==7:
                    
                    for i in range(7):
                       points.append([approx.ravel()[2*i], approx.ravel()[2*i+1]])

                    points.sort()
                   
                    minimum = points[1][0] - points[0][0]
                    maximum = points[6][0] - points[5][0]

                    if maximum < minimum :
                        left_count += 1
                    else:
                        right_count += 1
                    
                    cv2.drawContours(frame,[approx],0,(0,0,0),5)
                    
                
              
        if left_count>right_count and left_count > 10:
            f = open("./data/arrow.txt", 'w')
            print("left")
            f.write("left")
            
            TX_data_py2(serial_port, 26)
            time.sleep(1)
            TX_data_py2(serial_port, 43)
            time.sleep(1)
            TX_data_py2(serial_port, 21)
            exit(1)
            
        if left_count<right_count and right_count > 10:
            f = open("./data/arrow.txt", 'w')
            print("right")
            f.write("right")
            TX_data_py2(serial_port, 26)
            time.sleep(1)
            TX_data_py2(serial_port, 43)
            time.sleep(1)
            TX_data_py2(serial_port, 21
            exit(1)
           
           
        #cv2.imshow("Frame",frame)
        cv2.imshow("MASK",mask)
        cv2.waitKey(1)
        time.sleep(0.1)
        

        
       
       
    
    
    f.close()
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
