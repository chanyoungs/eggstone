import os
from preprocessor import preprocess
import matplotlib.pyplot as plt

def preprocess_savefig(root, progressbar, paths, params):
    
    title = ['Original', 'Filtered']
    progressbar["value"] = 0
    progressbar["maximum"] = len(paths)
    for p in range(len(paths)):
        fig = plt.figure(figsize=(10, 10))
        
        for n in range(2):
            imgs = preprocess(
                img_path=paths[p][n],
                params=params)

            for m in range(2):
                plt.subplot(2, 2, 2*n+m+1)
                if n == 0:
                    plt.title(title[m])
                plt.imshow(imgs[m])
                plt.xticks([]), plt.yticks([])
                            
        plt.savefig(os.path.join(root, "preprocessed", f"bean_{p+1}"))
        plt.close()
        progressbar["value"] += 1
        progressbar.update()

    progressbar["value"] = 0
    progressbar.update()
