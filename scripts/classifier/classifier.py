# -----------------------------------
# 1. Preload
# -----------------------------------

# Imports
import os
import numpy as np
import keras
from keras.models import load_model
from skimage.transform import resize
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt

from preprocessor import preprocess

# Session config to remove the CUDNN error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

class Predict():
    
    def __init__(self, model_name=None, model_type=None, p=None):
        # Load settings
        with open("settings.txt", "r") as f:
            settings = eval(f.read())

        if model_name is None:
            model_name = settings["model"]
        if model_type is None:
            model_type = settings["model_type"]
        self.params_name = settings["params"]
        
        # Load model - DONE
        # self.model = load_model(os.path.join("models", model_name, f"best_{model_type}_optimised.h5"))
        self.model = load_model(os.path.join("models", model_name, f"best_{model_type}.h5"))
                           
        # Load x_train_mean - Done
        self.pixel_mean = np.load(os.path.join("models", model_name, "pixel_mean.npy"))

        # Predict once for initialisation. For some reason, the first prediction always take more time
        self.model.predict(np.array([self.pixel_mean]))

        # Choose threshold - DONE
        if p is None:
            while True:
                p = input("\n\n\n\nChoose probability threshold between 0~1(The higher the thershold, the higher the standard of quality and will filter out more beans): ")
                try:
                    p = float(p)
                    if 0 <= p and p <= 1:
                        self.p = p
                        break
                    else:
                        print(f"\nPlease type a number between 0 and 1. (You typed '{p}'): ")
                except ValueError:
                    print(f"\nPlease type a number between 0 and 1. (You typed '{p}'): ")
        else:
            p = 0.5
            

    def predict(self, img, show_image=False, print_details=False, label=None):
        # -----------------------------------
        # 2. Predict
        # -----------------------------------

        # Preprocess image
        img_preprocessed = preprocess(img_np=img, params_path=os.path.join("params", self.params_name))

        # Resize image - Done
        img_resized = resize(img, self.pixel_mean.shape, anti_aliasing=True)
        
        # Subtract pixel mean - Done
        img_spm = img_resized - self.pixel_mean

        # Predict
        prob_pred = self.model.predict(np.array([img_spm]))[0]
        verdict = prob_pred[1] > self.p
        if print_details:
            print(f"Probability prediction: {prob_pred[0]: .4f}, Threshold probability: {self.p: .4f}, Verdict: {verdict}")
            if label is not None:
                if label == verdict:
                    print("Prediction correct :)")
                else:
                    print("Prediction incorrect :(")
        if show_image:
            plt.imshow(img)
            plt.show()
        return verdict
