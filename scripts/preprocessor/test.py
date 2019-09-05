import matplotlib.pyplot as plt
from preprocess import preprocess

img = plt.imread("../../raw_images/defective/1/Side1/1.jpg")
_, img2 = preprocess(input_np=img, dic_path="params/default.txt")
plt.imshow(img2)
plt.show()
