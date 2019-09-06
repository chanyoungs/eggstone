import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def bgr2rgb(bgr_img):
    b, g, r = cv2.split(bgr_img)       # get b,g,r
    return cv2.merge([r, g, b])     # switch it to rgb

def rgb2bgr(rgb_img):
    r, g, b = cv2.split(rgb_img)       # get b,g,r
    return cv2.merge([b, g, r])     # switch it to rgb

def preprocess(img_np=None, img_path="", params_path="", params={}, size_out=(350, 350)):
    if img_np is not None:
        img = rgb2bgr(img_np)
    elif img_path != "":
        img = cv2.imread(img_path)
    else:
        raise NameError('No input given. Please give an img_path or input_np.')

    if params_path != "":
        with open(params_path, 'r') as f:
            params = eval(f.read())
    elif params == {}:
        with open(os.path.join("params", "default.txt"), 'r') as f:
            params = eval(f.read())        
        
    height,width,depth = img.shape
    blurred_img = cv2.GaussianBlur(img, (params["kernel_size"], params["kernel_size"]), 0)
    hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    img_lum = 0.0722*blurred_img[:, :, 0] + 0.7152*blurred_img[:, :, 1] + 0.2126*blurred_img[:, :, 2] # Luminosity
    mask_lum = cv2.inRange(img_lum, params["lum_lower"], params["lum_upper"])
    mask_hsv = cv2.inRange(hsv, np.array(params["hsv_lower"], dtype='uint8'), np.array(params["hsv_upper"], dtype='uint8'))
    mask = (255 - mask_hsv) * mask_lum
    # mask = 255 - mask_hsv

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area

    # Create an empty canvas to draw the contour mask on
    mask2 = np.zeros((height,width), np.uint8)
    cv2.drawContours(mask2, [max_contour], -1, 1, cv2.FILLED)

    # Do AND operation between the original image and the created mask
    img_filtered = cv2.bitwise_and(img, img, mask = mask2)

    x,y,w,h = cv2.boundingRect(max_contour)
    # cv2.rectangle(img_filtered,(x,y),(x+w,y+h),(0,255,0),5)
    # cv2.putText(img_filtered, str((w, h)), (256, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

    w_out, h_out = size_out
    focused = np.zeros((w_out, h_out, 3), np.uint8)
    bean = img_filtered[y:y+h, x:x+w, :]
    if w > 450 or h > 450:
        print(bean.shape, focused[int(w_out/2)-int(w/2): int(w_out/2)+int(w/2)+1, int(h_out/2)-int(h/2): int(h_out/2)+int(h/2)+1].shape)
    focused[int(h_out/2)-int(h/2): int(h_out/2)+int(h/2)+h%2, int(w_out/2)-int(w/2): int(w_out/2)+int(w/2)+w%2] = bean
    # cv2.rectangle(focused, (0, 0),(450,450),(0,255,0),5)

    # # Find centre of contour
    # M = cv2.moments(max_contour)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    # cv2.circle(img_filtered, (cX, cY), 7, (0, 0, 255), -1)
    # l = 200
    # cv2.rectangle(img_filtered, (cX - l, cY - l), (cX + l, cY + l), (0, 255, 0), 5)

    # return bgr2rgb(img), bgr2rgb(img_filtered)
    return bgr2rgb(img), bgr2rgb(focused)
