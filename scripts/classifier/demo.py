import numpy as np
from classifier import Predict
import time
import sys
import os

setup = False
show_image = False
print_details = True

# Test run
print(os.getcwd(), __file__)
module_directory = os.path.dirname(os.path.abspath(__file__))
imgs = np.load(os.path.join(module_directory, "data_demo", "x_val.npy"))
labels = np.load(os.path.join(module_directory, "data_demo", "y_val.npy"))
test = Predict(setup)

while True:
    ind = int(input(f"Choose image index(0~{imgs.shape[0]}): "))
    start_time = time.time()
    test.predict(img=imgs[ind], show_image=show_image,
                 print_details=print_details, label=labels[ind])
    print(f"Prediction took {time.time() - start_time:.4f}s")
