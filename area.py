import numpy as np
import argparse
import cv2
import serial
import time
from serialdata import *


def loop(serial_port):
    W_View_size = 320  # original : 320
    H_View_size = int(W_View_size / 1.333)

    FPS = 10  # PI CAMERA: 320 x 240 = MAX 90

    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)

    f = open("./data/arrow.txt", 'r')
    direction = f.readline()
    print(direction)
    if direction == "left":
        TX_data_py2(serial_port, 28)
        TX_data_py2(serial_port, 31)
    elif direction == "right":
        TX_data_py2(serial_port, 30)
        TX_data_py2(serial_port, 31)

    lower_green = (35, 30, 30)
    upper_green = (100, 255, 255)

    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])
    time.sleep(5)
    while True:
        wait_receiving_exit()
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        safe_mask = cv2.inRange(hsv, lower_green, upper_green)

        # green = cv2.bitwise_and(frame, frame, mask = safe_mask)

        dan_mask = cv2.inRange(hsv, lower, upper)

        safe_count = len(hsv[np.where(safe_mask != 0)])
        dan_count = len(hsv[np.where(dan_mask != 0)])

        print("safe_count {}".format(safe_count))
        print("dan_count {}".format(dan_count))

        if safe_count > 15000 and dan_count < 100:  # original safe_count = 15000 , dan_count = 30
            print("safe_zone")
            f = open("./data/area.txt", 'w')
            f.write("safe")
            f.close()
            TX_data_py2(serial_port, 38)
            time.sleep(3)
            TX_data_py2(serial_port, 21)
            break

        elif dan_count > 15000 and safe_count < 100:  # original dan_count = 15000 , safe_count = 30
            print("dangerous_zone")
            f = open("./data/area.txt", 'w')
            f.write("dangerous")
            f.close()

            TX_data_py2(serial_port, 37)
            time.sleep(3)
            TX_data_py2(serial_port, 21)

            break

        cv2.imshow("safe_mask", safe_mask)
        cv2.imshow("dan_mask", dan_mask)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    time.sleep(1)
    exit(1)


if __name__ == '__main__':
    BPS = 4800  # 4800,9600,14400, 19200,28800, 57600, 115200

    serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
    serial_port.flush()  # serial cls

    serial_t = Thread(target=Receiving, args=(serial_port,))
    serial_t.daemon = True

    serial_d = Thread(target=loop, args=(serial_port,))
    serial_d.daemon = True

    print("start")
    serial_t.start()
    serial_d.start()

    # serial_t.join()
    serial_d.join()
    print("end")