# -----------------------------------
# 1. Preload
# -----------------------------------

# Imports
import sys
import os
import numpy as np
import keras
from keras.models import load_model
from skimage.transform import resize

class Predict():
    
    def __init__(self, model_name, model_type):
        # Load model - DONE
        self.model = load_model(os.path.join("models", model_name, f"best_{model_type}.h5"))
                           
        # Load x_train_mean - Done
        self.pixel_mean = np.load(os.path.join("models", model_name, "pixel_mean.npy"))

        # Choose threshold - DONE
        p = input("Choose probability threshold between 0~1(The higher the thershold, the higher the standard of quality and will filter out more beans): ")
        while True:
            if isinstance(p, float) and 0 <= p and p <= 1:
                break
            else:
                print(f"Please type a number between 0 and 1. (You typed '{p}'): ")

    def predict(img):
        # -----------------------------------
        # 2. Predict
        # -----------------------------------

        # Resize image - Done
        img_resized = resize(image, self.pixel_mean.shape, anti_aliasing=True)
        
        # Subtract pixel mean - Done
        img_spm = img_resized - self.pixel_mean

        # Predict ?
        return self.model.predict(img_spm) > p
