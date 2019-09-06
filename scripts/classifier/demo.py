import numpy as np
from classifier import Predict
import time
import sys
import os

# Test run
imgs = np.load(os.path.join("data_demo", "x_val.npy"))
labels = np.load(os.path.join("data_demo", "y_val.npy"))
test = Predict(setup=False)

while True:
    ind = int(input(f"Choose image index(0~{imgs.shape[0]}): "))
    start_time = time.time()
    test.predict(img=imgs[ind], show_image=True, print_details=True, label=labels[ind])
    print(f"Prediction took {time.time() - start_time:.4f}s")














