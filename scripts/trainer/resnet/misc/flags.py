import tensorflow as tf

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

def create_flags():
    flags = tf.app.flags
    FLAGS = tf.app.flags.FLAGS

    flags.DEFINE_string("model", "test", "m") # Used to create directories and filenames
    flags.DEFINE_string("data", "64", "d") # 54, 256
    flags.DEFINE_integer("n", 3, "n") # 3~
    flags.DEFINE_integer("version", 1, "v") # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    flags.DEFINE_integer("batch_size", 32, "b") # orig paper trained all networks with batch_size=128
    flags.DEFINE_integer("epochs", 200, "e")
    flags.DEFINE_boolean("data_augmentation", True, "da")
    flags.DEFINE_boolean("subtract_pixel_mean", True, "spm") # Subtracting pixel mean improves accuracy

    return FLAGS

print("Check")
