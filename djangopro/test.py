from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import PIL.Image as image
import cv2
import numpy as np
def cv_show(img, name='img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

noplant = cv2.imread('noplant.png')
row, col, z = np.shape(noplant)
imgHSV = cv2.imread('img.png')
imgHSV = cv2.cvtColor(imgHSV, cv2.COLOR_BGR2HSV)
pic_part = image.new("RGBA", (row, col))
print(row, col)
for i in range(row):
    for j in range(col):
        if (noplant[i][j] == [255, 255, 0]).all():
            pass
        else:
            if imgHSV[i][j][0] in range(30, 99) and imgHSV[i][j][1] in range(43, 255) and imgHSV[i][j][2] in range(46, 255):
                if imgHSV[i][j][2] in range(46, 255):
                    pic_part.putpixel((i, j), (0, 200, 0, 255))  # 绿色
                else:
                    pic_part.putpixel((i, j), (252, 230, 202, 255))  # 暖色调植物
            else:
                pic_part.putpixel((i, j), tuple(noplant[i][j]))
pic_part.save("result.png", "PNG")