import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# model = load_model(filepath_acc) 
model = load_model("./saved_models/defect_hunter_best_loss.h5")

x = np.load("data/images.npy")
y = np.load("data/labels.npy")

x_train, _, _, _ = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=33)
x_train_mean = np.mean(x_train, axis=0)


def run(n, init=False):
    img_pil = load_img(f"Images/good/Set04-good.01.{int(n):02}.jpg")
    start_time = time.time()
    img_pil.thumbnail((64, 64))
    img_pil = img_to_array(img_pil)
    x_sample = img_pil.astype('float32') / 255
    x_sample -= x_train_mean
    model.predict(x_sample.reshape(1, 64, 64, 3))
    time_elapsed = time.time() - start_time
    if not init:
        print(time_elapsed)

run(1, True)

while True:
    k = input("Type number 1~45 or type 'q' to quit: ")
    if k == 'q':
        break
    else:
        run(k)
