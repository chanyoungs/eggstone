import numpy as np
from classifier import Predict
import time
import sys

# model = sys.argv[1]

# Test run
imgs = np.load("../outputs/128/predictions/x_val.npy")
labels = np.load("../outputs/128/predictions/y_val.npy")
test = Predict()

while True:
    ind = int(input(f"Choose image index(0~{imgs.shape[0]}): "))
    start_time = time.time()
    test.predict(img=imgs[ind]/255, show_image=False, print_details=False)
    print(f"Prediction took {time.time() - start_time:.4f}s")














