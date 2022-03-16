from skimage import data, segmentation, color
import os
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2
def main(filename, slic_num=1000, compactness_num=10):
    img = cv2.imread(filename)
    labels1 = segmentation.slic(img, compactness=int(compactness_num), n_segments=int(slic_num))
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    cv2.imwrite(os.path.join(os.getcwd(), 'static/images', "slic_result.png"), out1)
    return out1