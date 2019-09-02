import os
print(os.getcwd())
from contour_detection_bean import filter_bean
from contour_detection_blue import filter_blue
from contour_detection_black import filter_black
import matplotlib.pyplot as plt

# Choose lower and upper colour range

# --bean--
# lower = (20.5, 10, 150)
# upper = (100, 255, 255)

# lower = (14.2, 25.5, 100)
# upper = (105, 142.1, 200)

# --blue--
lower = (70, 50, 0)
upper = (150, 110, 255)
kernel_size = 25

# --black--
# lower = 0
# upper = 90

root = "defective"
for set in range(5):
    for i in range(16):
        path_side1 = os.path.join(root, str(set+1), "Side1", f'{i+1}.jpg')
        path_side2 = os.path.join(root, str(set+1), "Side2", f'{i+1}.jpg')
        print(f'set_{set+1} image_{i+1}')
        img_side1, img_filtered_side1 = filter_blue(path_side1, lower=lower, upper=upper, kernel_size=kernel_size)
        img_side2, img_filtered_side2 = filter_blue(path_side2, lower=lower, upper=upper, kernel_size=kernel_size)

        fig = plt.figure(figsize=(10, 10))

        plt.subplot(221)
        plt.title('Original')
        plt.imshow(img_side1)
        plt.xticks([]), plt.yticks([])

        plt.subplot(222)
        plt.title('Filtered')
        plt.imshow(img_filtered_side1)
        plt.xticks([]), plt.yticks([])

        plt.subplot(223)
        plt.imshow(img_side2)
        plt.xticks([]), plt.yticks([])

        plt.subplot(224)
        plt.imshow(img_filtered_side2)
        plt.xticks([]), plt.yticks([])

        plt.savefig(os.path.join("preprocessed", f'set_{set+1} image_{i+1}'))
        plt.close()
