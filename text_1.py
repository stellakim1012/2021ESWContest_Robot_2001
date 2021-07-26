import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def Recog(img_color):
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 40])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_red = np.array([160, 120, 40])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    red_mask = mask0 + mask1

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    red_hsv = img_hsv.copy()
    blue_hsv = img_hsv.copy()

    red_count = len(red_hsv[np.where(red_mask != 0)])
    blue_count = len(blue_hsv[np.where(blue_mask != 0)])

    if red_count > blue_count:
        color = "red"
        red_hsv[np.where(red_mask != 0)] = 0
        red_hsv[np.where(red_mask == 0)] = 255

    else:
        color = "blue"
        blue_hsv[np.where(blue_mask != 0)] = 0
        blue_hsv[np.where(blue_mask == 0)] = 255

    print(color)

    return color

def color_img(img, push_color):
    if push_color == "red":
        lower = (0 - 10, 60, 60)
        upper = (0 + 10, 255, 255)

    # elif push_color == "green":
    #     lower = (60 - 10, 100, 100)
    #     upper = (60 + 10, 255, 255)
    #
    # elif push_color == "yellow":
    #     lower = (30 - 10, 100, 100)
    #     upper = (30 + 10, 255, 255)

    elif push_color == "blue":
        lower = (120 - 20, 60, 60)
        upper = (120 + 20, 255, 255)

    img_color = img

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    return img_result

while(True):
    ret, frame = cap.read()    # Read 결과와 frame

    if(ret) :
        color = Recog(frame)
        img_color = color_img(frame, color)
        copy = img_color.copy()
        gray = cv2.cvtColor(img_color,  cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)  # blue

            area = cv2.contourArea(cnt)

            x, y, w, h = cv2.boundingRect(cnt)

            filepath = ['./result/']

            if (area > 10000 and area < 20000) and (w/h > 0.8 and w/h < 1.2):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                filepath.append(str(area))
                filepath.append('.jpg')
                filepath = ''.join(filepath)

                crop_img = frame[y:y + h, x:x + w]

                cv2.imwrite(filepath, crop_img)

        cv2.imshow('result', frame)    # 컬러 화면 출력

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



# img_color = cv.imread('alphabet.jpg')
# img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
# ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
# contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# for cnt in contours:
#     cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 3)  # blue
#
#     area = cv.contourArea(cnt)
#     print(area)
#
#     x, y, w, h = cv.boundingRect(cnt)
#
#     filepath=['./']
#
#     if area > 5000 :
#         cv.rectangle(img_color, (x, y), (x +w, y+h), (36, 255, 12), 2)
#         filepath.append(str(area))
#         filepath.append('.jpg')
#         filepath = ''.join(filepath)
#
#         crop_img = img_color[y:y+h, x:x+w]
#
#         cv.imwrite(filepath, crop_img)
#
#
# cv.imshow("result", img_color)
#
# cv.waitKey(0)