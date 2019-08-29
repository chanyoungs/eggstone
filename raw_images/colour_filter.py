import numpy as np
import matplotlib.pyplot as plt
import cv2

def bgr2rgb(bgr_img):
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    return cv2.merge([r,g,b])     # switch it to rgb

img = cv2.imread('16369204-2019-08-24-153315.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([25, 40, 50])
upper = np.array([100, 255, 255])

mask = cv2.inRange(hsv, lower, upper)
bgr_img = cv2.bitwise_and(img, img, mask = mask)

plt.subplot(121)
plt.imshow(bgr2rgb(bgr_img))
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(bgr2rgb(img))
plt.xticks([]), plt.yticks([])

plt.show()
