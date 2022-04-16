from skimage import data, segmentation, color, io
import os
import numpy as np
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2
def main(filename, slic_num=1000, compactness_num=10):
    img = cv2.imread(filename)
    labels1 = segmentation.slic(img, compactness=int(compactness_num), n_segments=int(slic_num))
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    out2 = segmentation.mark_boundaries(out1, labels1, (0,0,0))

    cv2.imwrite(os.path.join(os.getcwd(), 'static/images', "slic_result.png"), out1)
    cv2.imwrite(os.path.join(os.getcwd(), 'static/images', "slic_boundaries.png"), out2*255)
    return out1
