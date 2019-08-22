import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# img_w = 256
# img_h = 256
img_w = 400
img_h = 400

def images_to_nparray(folder_path, error_log_file_name):
    file_names = os.listdir(folder_path)
    error_logs = ""

    n = len(file_names)

    # Original Dimensions
    img_w_original = 400
    img_h_original = 400
    n_channels = 3
    n_classes = 1

    # dataset = np.ndarray(shape=((n, n_channels, img_h, img_w)), dtype=np.float32)
    dataset_list = []



    i, percent, percent_temp, error_index = 0, 0, 0, 0
    for file_name in file_names:
        img_pil = load_img(os.path.join(folder_path, file_name))
        dim = np.array(img_pil).shape
    #     print(dim, (img_h_original, img_w_original, n_channels), dim == (img_h_original, img_w_original, n_channels))
        if not dim == (img_h_original, img_w_original, n_channels):
            error_index += 1
            error_logs += f"\n{error_index}. {file_name} has dimension {dim}"
            i += 1
        else:
            img_pil.thumbnail((img_w, img_h))

            dataset_list.append(img_to_array(img_pil))

            i += 1
            percent_temp = 100 * i / n
            if percent_temp > percent + 10:
                percent += 10
                print(f"{percent}% complete!")
    print("100% complete\n")

    dataset = np.array(dataset_list)
    print(f"Numpy array of images created with dimension: {dataset.shape}\n")

    if not error_logs == "":
        f = open(error_log_file_name, "w")
        f.write(error_logs)
        f.close()
        print(f"{error_index} error logs created in {error_log_file_name}\n")
        
    return dataset

good_imgs = images_to_nparray("../../images/good/", "error_logs_good.txt")
bad_imgs = images_to_nparray("../../images/bad/", "error_logs_bad.txt")

def create_x_and_y_train_data(dataset0, dataset1):
    if not dataset1[0].shape == dataset1[1].shape:
        print(f"Error: Dimensions of the datasets are inconsistent. {dataset1[0].shape} vs {dataset2[1].shape}")
    else:
        n = dataset0.shape[0] + dataset1.shape[0]
        
        X = np.ndarray(shape=(n, dataset0.shape[1], dataset0.shape[2], dataset0.shape[3]), dtype=np.float32)        
        X[:dataset0.shape[0]] = dataset0
        X[dataset0.shape[0]:] = dataset1
        
        y = np.zeros(n)
        y[dataset0.shape[0]:] = 1
        
    return X, y

X, y = create_x_and_y_train_data(good_imgs, bad_imgs)

np.save(f"../../data/images_{img_w}x{img_h}", X)
np.save(f"../../data/labels_{img_w}x{img_h}", y)
