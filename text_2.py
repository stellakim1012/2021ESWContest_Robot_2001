import cv2
import numpy as np
import pytesseract

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def Recog(img_color):
    img_crop = cv2.resize(img_color, (64, 64))

    #cv2.imshow('crop', img_crop)  # 화면 확인용

    img_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
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

    elif push_color == "blue":
        lower = (120 - 20, 60, 60)
        upper = (120 + 20, 255, 255)

    img_color = img

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    #cv2.imshow('mask', img_result)  # 화면 확인용

    return img_result

def textRecog(textimage):
    textimage = cv2.resize(textimage, (64, 64))
    textimage = cv2.cvtColor(textimage, cv2.COLOR_BGR2GRAY)

    result = np.zeros((64, 128), np.uint8) + 255
    result[:, :64] = textimage
    result[:, 64:128] = textimage

    cv2.imshow("canny", result)
    cv2.waitKey(1)

    text_image = pytesseract.image_to_string(result)
    text_image.replace(" ", "")
    text_image.rstrip()
    text_image = text_image[0:2]

    if text_image == "AA":
        text = "A"
    elif text_image == "BB":
        text = "B"
    elif text_image == "CC":
        text = "C"
    elif text_image == "DD":
        text = "D"
    else:
        text = "error"

    if text == "error":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        textimage = cv2.dilate(textimage, kernel)

        result = np.zeros((64, 128), np.uint8) + 255
        result[:, :64] = textimage
        result[:, 64:128] = textimage

        cv2.imshow("canny", result)
        cv2.waitKey(1)

        text_image = pytesseract.image_to_string(result, lang='eng')
        text_image.replace(" ", "")
        text_image.rstrip()
        text_image = text_image[0:2]

        if text_image == "AA":
            text = "A"
        elif text_image == "BB":
            text = "B"
        elif text_image == "CC":
            text = "C"
        elif text_image == "DD":
            text = "D"
        else:
            text = "error"

    if text == "error":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        textimage = cv2.erode(textimage, kernel)

        result = np.zeros((64, 128), np.uint8) + 255
        result[:, :64] = textimage
        result[:, 64:128] = textimage

        cv2.imshow("canny", result)
        cv2.waitKey(1)

        text_image = pytesseract.image_to_string(result, lang='eng')
        text_image.replace(" ", "")
        text_image.rstrip()
        text_image = text_image[0:2]

        if text_image == "AA":
            text = "A"
        elif text_image == "BB":
            text = "B"
        elif text_image == "CC":
            text = "C"
        elif text_image == "DD":
            text = "D"
        else:
            text = "error"

    return text

while(True):
    ret, frame = cap.read()

    if(ret) :
        color = Recog(frame)
        img_color = color_img(frame, color)
        copy = img_color.copy()
        gray = cv2.cvtColor(img_color,  cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv2.drawContours(frame, [cnt], 0, (255, 255, 0), 3)

            area = cv2.contourArea(cnt)

            x, y, w, h = cv2.boundingRect(cnt)

            #filepath = ['./result/']

            if (area > 10000 and area < 20000) and (w/h > 0.8 and w/h < 1.2):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                # filepath.append(str(area))
                # filepath.append('.jpg')
                # filepath = ''.join(filepath)

                crop_img = frame[y:y + h, x:x + w]

                cv2.imshow('crop_alphabet', crop_img)

                text = textRecog(crop_img)
                print(text)

                # cv2.imwrite(filepath, crop_img)

        cv2.imshow('result', frame)    # 컬러 화면 출력

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()