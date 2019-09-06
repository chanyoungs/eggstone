import os
import numpy as np
from preprocessor import preprocess
from PIL import Image

def preprocess_savefig(root, progressbar, paths, params, type):

    if not os.path.isdir(os.path.join(root, "preprocessed")):
        for x in ["individual", "joined"]:
            for y in ["healthy", "defective"]:
                os.makedirs(os.path.join(root, "preprocessed", x, y))
    
    progressbar["value"] = 0
    progressbar["maximum"] = len(paths)

    _, img_sample = preprocess(img_path=paths[0][0], params=params)
    shape = img_sample.shape
    imgs = []
    for p in range(len(paths)):
        img_join = np.zeros((shape[0]*2, shape[1], shape[2]), dtype='uint8')
        for n in range(2):
            _, img_np = preprocess(img_path=paths[p][n], params=params)
            img_join[n*shape[0]: (n+1)*shape[0]] = img_np
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(root, "preprocessed", "individual", type, f"bean{p+1}_side{n+1}.png"))
        img_pil = Image.fromarray(img_join)
        img_pil.save(os.path.join(root, "preprocessed", "joined", type, f"bean{p+1}.png"))
        progressbar["value"] += 1
        progressbar.update()
    progressbar["value"] = 0
    progressbar.update()

# def preprocess_savefig(root, progressbar, paths, params, type):
    
#     title = ['Original', 'Filtered']
#     progressbar["value"] = 0
#     progressbar["maximum"] = len(paths)
#     for p in range(len(paths)):
#         fig = plt.figure(figsize=(10, 10))
        
#         for n in range(2):
#             imgs = preprocess(
#                 img_path=paths[p][n],
#                 params=params)

#             for m in range(2):
#                 plt.subplot(2, 2, 2*n+m+1)
#                 if n == 0:
#                     plt.title(title[m])
#                 plt.imshow(imgs[m])
#                 plt.xticks([]), plt.yticks([])
                            
#         plt.savefig(os.path.join(root, "preprocessed", type, f"bean_{p+1}"))
#         plt.close()
#         progressbar["value"] += 1
#         progressbar.update()

#     progressbar["value"] = 0
#     progressbar.update()
