import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def IOUCalculation(actual, predicted, num):
    actual_count = [0 for v in range(num)]
    pred_count = [0 for v in range(num)]
    h, w = np.shape(actual)
    for i in range(h):
        for j in range(w):
            actual_count[actual[i][j]] += 1
            pred_count[predicted[i][j]] += 1
    I = np.diagonal(fast_hist(np.array(actualArr), np.array(predictedArr), num))
    U = np.array(actual_count) + np.array(pred_count) - I
    IOU = I / U
    return IOU
def fast_hist(a, b, n):
    """
    生成混淆矩阵
    a 是形状为(HxW,)的预测值
    b 是形状为(HxW,)的真实值
    n 是类别数
    """
    # 确保a和b在0~n-1的范围内，k是(HxW,)的True和False数列
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
def loadPicture(pic1):
    image1 = np.copy(Image.open(pic1))
    h, w, z = np.shape(image1)
    print(h, w)
    image = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if (image1[i][j] == [255, 255, 0]).all():
                image[i][j] = 0    # 黄
            elif (image1[i][j] == [0, 255, 255]).all():
                image[i][j] = 1 # 青
            elif (image1[i][j] == [0, 0, 255]).all():
                image[i][j] = 2  # 蓝
            else:
                image[i][j] = 3  # 绿
    print(image)
    return image

predictedArr = loadPicture("result.png")
actualArr = np.copy(Image.open("label.png"))
print(actualArr)
# actualArr = [
#     [0, 0, 0, 0],
#     [0, 1, 1, 4],
#     [5, 5, 2, 4],
#     [5, 3, 3, 4]
# ]
# predictedArr = [
#     [0, 0, 0, 0],
#     [0, 0, 1, 1],
#     [5, 5, 2, 4],
#     [5, 3, 3, 4]
# ]
# 每个类的重叠度
iou = IOUCalculation(actualArr, predictedArr, 3)
# 平均重叠度
miou = np.nanmean(iou)
print(iou)
print(miou)
