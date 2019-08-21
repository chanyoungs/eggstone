from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Session config to remove the CUDNN error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Flags
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("model", "test", "m") # Used to create directories and filenames
flags.DEFINE_string("data", "64", "d") # 54, 256
flags.DEFINE_integer("n", 3, "n") # 3~
flags.DEFINE_integer("version", 1, "v") # 1,2,3
flags.DEFINE_integer("batch_size", 32, "b")
flags.DEFINE_integer("epochs", 200, "e")
flags.DEFINE_boolean("data_augmentation", True, "da")
flags.DEFINE_boolean("subtract_pixel_mean", True, "spm")

# Load data
x = np.load(f"data/images_{FLAGS.data}x{FLAGS.data}.npy")
y = np.load(f"data/labels_{FLAGS.data}x{FLAGS.data}.npy")

# Make directories
path = os.path.join("outputs", FLAGS.model)
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

#Shuffling and splitting data into train, validation, test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=33)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=33)
np.save(os.path.join(path, "predictions", "x_val"), x_val)
np.save(os.path.join(path, "predictions", "y_val"), y_val)
print(x_val[0])
plt.imshow(x_val[0])
plt.show()

#@title Resnet reference: https://keras.io/examples/cifar10_resnet/

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model


# Training parameters
batch_size = FLAGS.batch_size  # orig paper trained all networks with batch_size=128
epochs = FLAGS.epochs
data_augmentation = FLAGS.data_augmentation
num_classes = 2

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = FLAGS.subtract_pixel_mean

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = FLAGS.n

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = FLAGS.version


# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_val -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
#     print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=2):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(path, 'checkpoints')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath_acc = os.path.join(save_dir, "best_acc.h5")
filepath_loss = os.path.join(save_dir, "best_loss.h5")

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint_acc = ModelCheckpoint(filepath=filepath_acc,
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True)

checkpoint_loss = ModelCheckpoint(filepath=filepath_loss,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6,
                               verbose=0)

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.accuracies = []
        self.val_accuracies = []
        self.losses = []
        self.val_losses = []
        self.val_aucs = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Calculate AUC
        y_pred = model.predict(x_val)
        np.save(os.path.join(path, "predictions", "y_pred"), y_pred[:, 1])
        y_compare = np.zeros((y_val.shape[0], 2))
        y_compare[:, 0] = y_pred[:, 1]
        y_compare[:, 1] = y_val[:, 1]
        y_compare_df = pd.DataFrame(data=y_compare, columns=["Predicted Probabilities", "Ground Truth"])
        y_compare_df.to_csv(os.path.join(path, "predictions", f"comparisons_{epoch}.csv"), index=None, header=True)
        
        fpr, tpr, thresholds = roc_curve(y_val[:, 1], y_pred[:, 1])
        auc_value = auc(fpr, tpr)
        self.val_aucs.append(auc_value)

        # Plot figures
        fig = plt.figure(figsize=(20, 20))
        fig.patch.set_facecolor('white')
        plt.suptitle(f'Accuracies, Losses, ROC and AUC epoch({epoch})')

        # Accuracies and losses
        gs = gridspec.GridSpec(2, 2)
        plt.subplot(gs[0, :])
        plt.title(f"Accuracies, losses and AUC")
        plt.plot(self.x, self.losses,
                 label=f"loss {logs.get('loss'): .4}")
        plt.plot(self.x, self.val_losses,
                 label=f"val_loss {logs.get('val_loss'): .4}")
        plt.plot(self.x, self.accuracies,
                 label=f"acc {logs.get('acc'): .4}")
        plt.plot(self.x, self.val_accuracies,
                 label=f"val_acc {logs.get('val_acc'): .4}")
        plt.plot(self.x, self.val_aucs,
                 label=f"val_auc {auc_value: .4}")
        
        y_loss_m, y_acc_M, y_auc_M = min(self.val_losses), max(self.val_accuracies), max(self.val_aucs)
        x_loss_m, x_acc_M, x_auc_M = self.val_losses.index(y_loss_m), self.val_accuracies.index(y_acc_M), self.val_aucs.index(y_auc_M)

        plt.plot(x_loss_m, y_loss_m ,'o')
        plt.annotate(f"Min val_loss: {y_loss_m: .4}", xy=(x_loss_m, y_loss_m))
        plt.plot(x_acc_M, y_acc_M ,'o')
        plt.annotate(f"Max val_acc: {y_acc_M: .4}", xy=(x_acc_M, y_acc_M))
        plt.plot(x_auc_M, y_auc_M ,'o')
        plt.annotate(f"Max val_auc: {y_auc_M: .4}", xy=(x_auc_M, y_auc_M))
        plt.legend()
        
        # ROC Curve
        plt.subplot(gs[1, 0])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        plt.subplot(gs[1, 1])
        plt.xlim(0, 0.3)
        plt.ylim(0.7, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        fig.savefig(os.path.join(path, "figures", "snapshots", f"figures{epoch}"))
        fig.savefig(os.path.join(path, "figures", "figures"))
        
plot_losses = PlotLosses()

callbacks = [checkpoint_acc, checkpoint_loss, lr_reducer, lr_scheduler, plot_losses]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_val, y_val),
                        epochs=epochs, steps_per_epoch=len(x_train)/batch_size,
                        verbose=1, workers=4, callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

from sklearn.metrics import roc_curve, auc

y_pred = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
auc_value = auc(fpr, tpr)

fig1 = plt.figure()
fig1.patch.set_facecolor('white')
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

fig2 = plt.figure()
fig2.patch.set_facecolor('white')
plt.xlim(0, 0.3)
plt.ylim(0.7, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Area = {:.3f}'.format(auc_value))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


no_steps = 10000.
for p in range(1, 20):
    y_pred_thresh = y_pred[:, 1] > p/no_steps
    cm = confusion_matrix(y_test[:, 1], y_pred_thresh)
    plot_confusion_matrix(cm=cm, target_names=["Healthy", "Defective"], normalize=False)
    negatives_array = y_pred_thresh == 0
    false_array = y_pred_thresh != y_test[:,1]
    false_negatives_array = negatives_array * false_array
    no_negatives = negatives_array.sum()
    no_false_negatives = false_negatives_array.sum()
    print(f"Threshold p = {p/no_steps}: {no_false_negatives}/{no_negatives} = {no_false_negatives/no_negatives * 100: .4}%")
