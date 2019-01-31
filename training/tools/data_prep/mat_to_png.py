# convert the mat files in SBD into a 
import os
import scipy
from scipy.io import loadmat
import numpy as np
import pdb

cls_dir = '/media/ssd1/austin/datasets/SBD/benchmark_RELEASE/dataset/cls'
out_dir = '/media/ssd1/austin/datasets/SBD/benchmark_RELEASE/dataset/cls_png'

for f in os.listdir(cls_dir):
    # open the .mat file
    mat_path = os.path.join(cls_dir, f)
    mat_obj = loadmat(mat_path)
    arr_comp_mats = mat_obj['GTcls'][0][0][0]
    assert(arr_comp_mats.shape == (20, 1))
    img = np.zeros(mat_obj['GTcls'][0][0][0][0][0].shape).astype(np.uint8)
    # for each class, there is a compressed matrix 
    for i, comp_mat_arr in enumerate(arr_comp_mats):
        assert(comp_mat_arr.shape == (1,))
        comp_mat = comp_mat_arr[0]
        if np.all(comp_mat == np.zeros(comp_mat.shape)):
            continue
        arr = comp_mat.toarray()
        img[np.where(arr != 0)] = i+1 # replace 1s with the class number
    assert(not np.all(img == np.zeros(img.shape)))
    img_id = f[:-4] # remove .mat ext
    scipy.misc.imsave(os.path.join(out_dir, img_id + '.png'), img)
    print(img_id)
    
