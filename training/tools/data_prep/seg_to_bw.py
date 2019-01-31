# tested with python3

import os
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
from scipy import misc
import numpy as np
import xml.etree.ElementTree as ET
import pdb

SegClassPath = '/home/austin/VOC2012/VOCdevkit_18-May-2011/VOC2012/SegmentationClass'
AnnoPath = '/home/austin/VOC2012/VOCdevkit_18-May-2011/VOC2012/Annotations'
OutPath = '/home/austin/VOC2012/VOCdevkit_18-May-2011/VOC2012/AugSegClass'

def chunks(l, n):
    # l is the list, n is the max chunk size
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

def find_colors():
    all_colors = np.array([[0,0,0]])
    counter = 0
    for f in os.listdir(SegClassPath):
        arr = misc.imread(os.path.join(SegClassPath, f))
        flat = arr.reshape(arr.shape[0]*arr.shape[1], 3)
        colors, inds = np.unique(flat, True, axis=0)
        all_colors = np.append(all_colors, colors, axis=0)
        if counter % 100:
            all_colors = np.unique(all_colors, axis=0)
            

    all_colors = np.unique(all_colors, axis=0)
    return all_colors

def map_color_to_img(all_colors):
    # map each color to an image that has only 3 colors (bg, ambig, and that class)
    color_to_img = {}
    bg = np.array([0, 0, 0], dtype=np.uint8)
    ambig = np.array([224, 224, 192], dtype=np.uint8)
    found_colors = [bg, ambig]

    for f in os.listdir(SegClassPath):
        arr = misc.imread(os.path.join(SegClassPath, f))
        flat = arr.reshape(arr.shape[0]*arr.shape[1], 3)
        colors, inds = np.unique(flat, True, axis=0)
        if len(colors) != 3:
            # ensure the image has only bg, ambig, and one class
            continue

        # find the interesting color
        interesting_color = None
        for c in colors:
            if np.all(c == bg): continue
            if np.all(c == ambig): continue
            interesting_color = c

        # determine if found already
        found_already = False
        for color in found_colors:
            if np.all(color == interesting_color):
                found_already = True

        if not found_already:
            found_colors.append(interesting_color)
            color_to_img[tuple(list(interesting_color))] = f
            
        if len(found_colors) == len(all_colors):
            return color_to_img
    print('ERROR: couldnt find all colors')
    return color_to_img

def find_colors_p(ncpus):
    # for finding colors in parallel
    # script finished before I finished writing this
    # ncpus is approximate
    # not tested
    fnames = [name for name in os.listdir(SegClassPath)]
    list_img_paths = chunks(fnames, len(fnames) // ncpus)
    pool = Pool(ncpus)
    out = pool.map(find_colors_h, list_img_paths)
    pool.join()
    pdb.set_trace()

def find_colors_h(img_paths):
    all_colors = np.array([[0,0,0]])
    counter = 0
    for f in img_paths:
        arr = misc.imread(os.path.join(SegClassPath, f))
        flat = arr.reshape(arr.shape[0]*arr.shape[1], 3)
        colors, inds = np.unique(flat, True, axis=0)
        all_colors = np.append(all_colors, colors, axis=0)
        if counter % 100:
            all_colors = np.unique(all_colors, axis=0)
            

    all_colors = np.unique(all_colors, axis=0)
    return all_colors

def show_colors(colors):
    for color in colors:
        print(color)
        img = np.reshape(color, (1,1,3)).astype(np.uint8)
        plt.imshow(img)
        plt.waitforbuttonpress(0) # this will wait for indefinite time

def convert_images(color_to_num):
    for f in os.listdir(SegClassPath):
        oimg = misc.imread(os.path.join(SegClassPath, f))
        new_img = np.zeros(oimg.shape[:2])
        for color in color_to_num:
            mask = np.all(oimg == color, axis=-1)
            new_img[mask] = color_to_num[color]
        new_img = new_img.astype(np.uint8)
        save_path = os.path.join(OutPath, f)
        misc.toimage(new_img).save(save_path)



if __name__ == "__main__":

    cpus = 6

    # find every unique color in the images, should be 22: 20 classes, bg, and ambiguous
    if os.path.isfile('all_colors.p'):
        with open('all_colors.p', 'rb') as f:
            all_colors = pickle.load(f)
    else:
        # all_colors = find_colors_p(4)
        # parallel _p not tested
        all_colors = find_colors()
        with open('all_colors.p', 'wb') as f:
            pickle.dump(all_colors, f)
    
    if os.path.isfile('c2i.p'):
        with open('c2i.p', 'rb') as f:
            color_to_img = pickle.load(f)
    else:
        color_to_img = map_color_to_img(all_colors)
        with open('c2i.p', 'wb') as f:
            pickle.dump(color_to_img, f)

    # now use the annotation xmls to assign each color a name
    color_to_name = {(0,0,0): 'background', (224,224,192): 'ambiguous'}

    for color,iname in color_to_img.items():
        aname = iname.replace('png', 'xml')
        anno_path = os.path.join(AnnoPath, aname) 

        # ensure only one name per image
        names = set()
        # parse the annotation xml file
        tree = ET.parse(anno_path)
        root = tree.getroot()
        for obj_elem in root.findall('object'):
            assert(len(obj_elem.findall('name')) == 1)
            names.add(obj_elem.findall('name')[0].text)
        assert(len(names) ==1)
        color_to_name[color] = names.pop()

    # use classes.txt, newline delimited file of alphabetically ordered class names,
    # to create a map from current color to new color
    with open('classes.txt', 'r') as f:
        classes = f.readlines()

    # IMPORTANT: labeling ambiguous as background
    name_to_num = {'background': 0, 'ambiguous': 0}
    for i, cl in enumerate(classes):
        cl = cl.strip()
        name_to_num[cl] = i # classes.txt includes background at first line

    # map old RGB colors to new B&W color (alphabetical number)
    color_to_num = {}
    for color in all_colors:
        color_to_num[tuple(list(color))] = name_to_num[color_to_name[tuple(list(color))]]

    # sofa looks all good

    # convert each RGB image to B&W image
    convert_images(color_to_num)

    # show all colors
    # show_colors(all_colors)
