# convert the mat files in SBD into a 
import os
import scipy
from scipy.io import loadmat
import numpy as np
import cv2
import pdb

cls_dir = '/media/ssd1/austin/datasets/SBD/benchmark_RELEASE/dataset/cls'
out_dir = '/media/ssd1/austin/datasets/SBD/benchmark_RELEASE/dataset/cls_png'

for f in os.listdir(cls_dir):
    # open the .mat file
    mat_path = os.path.join(cls_dir, f)
    mat_obj = loadmat(mat_path)
    img = mat_obj['GTcls']['Segmentation'][0][0]
    assert(not np.all(img == np.zeros(img.shape)))
    img_id = f[:-4] # remove .mat ext
    
    # new_img = img * 12
    # cv2.imshow(img_id, new_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    scipy.misc.imsave(os.path.join(out_dir, img_id + '.png'), img)
    print(img_id)
    
