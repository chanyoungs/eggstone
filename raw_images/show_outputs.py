import numpy as np
import glob
from contour_detection_bean import filter_bean
from contour_detection_blue import filter_blue
from contour_detection_black import filter_black
import matplotlib.pyplot as plt
import time

paths = glob.glob("*.jpg")
n = len(paths)

# Choose lower and upper colour range

# --bean--
# lower = (20.5, 10, 150)
# upper = (100, 255, 255)

# lower = (14.2, 25.5, 100)
# upper = (105, 142.1, 200)

# --blue--
# lower = (100, 0, 0)
# upper = (255, 255, 255)

# --black--
lower = 0
upper = 90


plt.figure(figsize=(10, 20))
for s in range(n):

    start_time = time.time()
    # img, img_filtered = filter_bean(paths[s], lower=lower, upper=upper)
    # img, img_filtered = filter_blue(paths[s], lower=lower, upper=upper)
    img, img_filtered = filter_black(paths[s], lower=lower, upper=upper)
    print(f'{time.time() - start_time:.4f}')
    
    plt.subplot(n, 2, 2*s+1)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])

    plt.subplot(n, 2, 2*s+2)
    plt.imshow(img_filtered)
    plt.xticks([]), plt.yticks([])

plt.show()
