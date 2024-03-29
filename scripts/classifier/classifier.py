# -----------------------------------
# 1. Preload
# -----------------------------------

# Imports
import os
import numpy as np
from keras.models import load_model
from skimage.transform import resize
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt

from preprocessor import preprocess
from setup import run_setup

# Session config to remove the CUDNN error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
# set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)
# set_session(sess)

module_directory = os.path.dirname(os.path.abspath(__file__))


class Predict():
    def __init__(self, setup=False):
        settings = os.path.join(module_directory, "settings.txt")
        if not os.path.isfile(settings) or setup:
            run_setup()
        # Load settings
        with open(settings, "r") as f:
            settings = eval(f.read())
        self.params_name = settings["params"]
        self.p_threshold = settings["p_threshold"]

        # Load x_train_mean - Done
        self.pixel_mean = np.load(os.path.join(module_directory,
                                               "models", settings["model"], "pixel_mean.npy"))

        print("Initialising model...")
        # Load model - DONE
        # self.model = load_model(os.path.join("models", settings["model"], f"best_{settings['model_type']}_optimised.h5"))
        self.model = load_model(os.path.join(module_directory,
                                             "models", settings["model"], f"best_{settings['model_type']}.h5"))

        # Predict once for initialisation. For some reason, the first prediction always take more time
        self.model.predict(np.array([self.pixel_mean]))
        print("Done!")

    def predict(self, img, show_image=False, print_details=False, label=None):
        # -----------------------------------
        # 2. Predict
        # -----------------------------------

        # Convert image values between 0 and 1
        img /= 255.

        # Preprocess image
        # _, img_preprocessed = preprocess(
        #     img_np=img, params_path=os.path.join(module_directory, "params", self.params_name))
        img_preprocessed = img

        # Resize image - Done
        # img_resized = resize(img, self.pixel_mean.shape, anti_aliasing=True)
        img_resized = resize(
            img_preprocessed, self.pixel_mean.shape, anti_aliasing=True)

        # Subtract pixel mean - Done
        img_spm = img_resized - self.pixel_mean

        # Predict
        prob_pred = self.model.predict(np.array([img_spm]))[0]
        verdict = prob_pred[1] > self.p_threshold
        if print_details:
            print(
                f"Probability prediction: {prob_pred[1]: .4f}, Threshold probability: {self.p_threshold: .4f}, Verdict: {verdict}")
            if label is not None:
                if label == verdict:
                    print("Prediction correct :)")
                else:
                    print("Prediction incorrect :(")
        if show_image:
            plt.imshow(img)
            plt.axis("off")
            if label is not None:
                plt.title(label)
            plt.show()
        return verdict
