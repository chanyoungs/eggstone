import numpy as np
import matplotlib.pyplot as plt
import cv2

def bgr2rgb(bgr_img):
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    return cv2.merge([r,g,b])     # switch it to rgb

def filter_black(path, lower=0, upper=50):
    img = cv2.imread(path)
    height,width,depth = img.shape
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, int(lower), int(upper))
    print(mask)

    _, contours, _ = cv2.findContours(255 - mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area

    # Create an empty canvas to draw an ellipse mask on
    mask2 = np.zeros((height,width), np.uint8)

    # smooth_contour = cv2.approxPolyDP(curve=max_contour, epsilon=5, closed=True)
    # cv2.drawContours(mask2, [smooth_contour], -1, 1, cv2.FILLED)
    # blurred_mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
    cv2.drawContours(mask2, [max_contour], -1, 1, cv2.FILLED)

    # Do AND operation between the original image and the created mask
    # img_filtered = cv2.bitwise_and(img, img, mask = mask2)
    img_filtered = cv2.bitwise_and(img, img, mask = mask2)
    
    return bgr2rgb(img), bgr2rgb(img_filtered)


    # plt.subplot(121)
    # plt.imshow(bgr2rgb(img))
    # plt.xticks([]), plt.yticks([])

    # plt.subplot(122)
    # plt.imshow(bgr2rgb(img_filtered))
    # plt.xticks([]), plt.yticks([])
    # plt.show()
