import numpy as np
from classifier import Predict
import time

# Test run
imgs = np.load("../../outputs/data340_v1_bs16/predictions/x_val.npy")
labels = np.load("../../outputs/data340_v1_bs16/predictions/y_val.npy")
test = Predict(model_name="340", model_type="loss")

while True:
    ind = int(input(f"Choose image index(0~{imgs.shape[0]}): "))
    start_time = time.time()
    test.predict(img=imgs[ind]/255, show_image=True, print_details=True, label=labels[ind])
    print(f"Prediction took {time.time() - start_time:.4f}s")
