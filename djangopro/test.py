from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2
def cv_show(img, name='img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img = cv2.imread('3.png', 0)

cv2.imwrite("ggg.png", img)