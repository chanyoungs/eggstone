import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(root, path, data):
    # Load data
    x = np.load(os.path.join(root, "data", f"images_{data}.npy"))
    y = np.load(os.path.join(root, "data", f"labels_{data}.npy"))

    # Shuffling and splitting data into train, validation, test sets
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=33)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=33)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=33)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=33)

    # Save validation set
    np.save(os.path.join(path, "predictions", "x_val"), x_val)
    np.save(os.path.join(path, "predictions", "y_val"), y_val)

    return [x_train, y_train, x_val, y_val, x_test, y_test]
