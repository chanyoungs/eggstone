import numpy as np
import matplotlib.pyplot as plt
import cv2

def bgr2rgb(bgr_img):
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    return cv2.merge([r,g,b])     # switch it to rgb

def filter_bean(path, lower=(25, 40, 50), upper=(100, 255, 255)):
    img = cv2.imread(path)
    height,width,depth = img.shape
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(lower, dtype='uint8'), np.array(upper, dtype='uint8'))

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area

    # Fit the closest ellipse
    ellipse = cv2.fitEllipse(max_contour)
    # Create an empty canvas to draw an ellipse mask on
    ellipse_img = np.zeros((height,width), np.uint8)
    # Draw the filled ellipse on the canvas
    cv2.ellipse(ellipse_img, ellipse, 255, cv2.FILLED)

    # cv2.drawContours(img, [max_contour], -1, (0, 255, 0), cv2.FILLED)

    # Do AND operation between the original image and the created mask
    img_filtered = cv2.bitwise_and(img, img, mask = ellipse_img)

    # Draw the ellipse on top of the orignal image for reference
    cv2.ellipse(img, ellipse, [0, 255, 0], 3)

    return bgr2rgb(img), bgr2rgb(img_filtered)

    # smooth_contour = cv2.approxPolyDP(curve=max_contour, epsilon=8, closed=True)
    # cv2.drawContours(img, [smooth_contour], -1, (0, 255, 0), cv2.FILLED)

    # plt.subplot(121)
    # plt.imshow(bgr2rgb(img))
    # plt.xticks([]), plt.yticks([])

    # plt.subplot(122)
    # plt.imshow(bgr2rgb(img_filtered))
    # plt.xticks([]), plt.yticks([])
    # plt.show()
