import os
from contour_detection_bean import filter_bean
from contour_detection_blue import filter_blue
from contour_detection_black import filter_black
import matplotlib.pyplot as plt

def preprocess_savefig(
        root,
        progressbar,
        paths,
        hsv_lower=(70, 50, 0),
        hsv_upper=(150, 110, 255),
        lum_lower=0,
        lum_upper=255,
        kernel_size=21):
    
    title = ['Original', 'Filtered']
    progressbar["value"] = 0
    progressbar["maximum"] = len(paths)
    for p in range(len(paths)):
        fig = plt.figure(figsize=(10, 10))
        
        for n in range(2):
            imgs = filter_blue(
                paths[p][n],
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                lum_lower=lum_lower,
                lum_upper=lum_upper,
                kernel_size=kernel_size)

            for m in range(2):
                plt.subplot(2, 2, 2*n+m+1)
                if n == 0:
                    plt.title(title[m])
                plt.imshow(imgs[m])
                plt.xticks([]), plt.yticks([])
                            
        plt.savefig(os.path.join(root, "preprocessed", f'bean_{p+1}'))
        plt.close()
        progressbar["value"] += 1
        progressbar.update()

    progressbar["value"] = 0
    progressbar.update()
