from __future__ import print_function
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from load_data import load_data
from model_resnet import run_resnet
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Session config to remove the CUDNN error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Get flags
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# # Model parameter
# # ----------------------------------------------------------------------------
# #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
# #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# # ----------------------------------------------------------------------------
# # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# # ---------------------------------------------------------------------------
flags.DEFINE_string("model", "test", "Used to create directories and filenames")
flags.DEFINE_string("data", "64", "Input dimensions")
flags.DEFINE_integer("n", 3, "Number of layers")
flags.DEFINE_integer("version", 1, "Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)")
flags.DEFINE_integer("batch_size", 32, "orig paper trained all networks with batch_size=128")
flags.DEFINE_integer("epochs", 200, "Total number of epochs")
flags.DEFINE_boolean("data_augmentation", True, "If set true, does data augmentation")
flags.DEFINE_boolean("subtract_pixel_mean", True, "Subtracting pixel mean improves accuracy")


# Define paths
root = "../../../"
path = os.path.join(root, "outputs", FLAGS.model)


# Make directories
directories = [
    ['checkpoints'],
    ['predictions'],
    ['figures', 'snapshots'],
]

for paths in directories:
    full_dir = path
    for folder in paths:
        full_dir = os.path.join(full_dir, folder)

    if not os.path.isdir(full_dir):
        print(f'{full_dir} does not exist. Creating path...')
        os.makedirs(full_dir)
    else:
        print(f'{full_dir} already exists')


# Load data
data = load_data(root, path, FLAGS.data)


# Train model
run_resnet(FLAGS.batch_size,
           FLAGS.epochs,
           FLAGS.data_augmentation,
           FLAGS.subtract_pixel_mean,
           FLAGS.n,
           FLAGS.version,
           path,
           data)

