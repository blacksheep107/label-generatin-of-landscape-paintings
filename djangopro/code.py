import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image as image
import cv2
import time
def distance(p1, p2):
    '''
    :param p1: 像素点
    :param p2: 像素点
    :return: 两点之间的欧式距离
    '''
    return np.sqrt(np.sum(np.square(p1 - p2)))
def loadPicture(filePath):
    '''
    处理图片
    :param filePath: 图片路径
    :return: 二维数组和图片高宽
    '''
    img = cv2.imread(filePath)
    h, w, z = np.shape(img)
    img = cv2.resize(img, (int(w / 2), int(h / 2)))
    print(h, w)
    h, w, z = np.shape(img)
    # 中值滤波
    img_median = cv2.medianBlur(img, 5)
    data = []
    for i in range(h):
        for j in range(w):
            # 归一化
            # data.append([img_median[i][j][0] / 256.0, img_median[i][j][1] / 256.0, img_median[i][j][2] / 256.0])
            data.append([img_median[i][j][0], img_median[i][j][1], img_median[i][j][2]])
    return np.array(data), h, w
# def getCent(data, k):
#     '''
#     kmeans++初始化聚类中心
#     :param data: 样本
#     :param k: 聚类中心个数
#     :return: 初始化后的聚类中心
#     '''
#     m, n = np.shape(data)   # m个点，一个点n维
#     cluster_centers = np.mat(np.zeros((k, n)))
#     # 随机选择一个样本点作为第一个聚类中心
#     index = np.random.randint(0, m)
#     cluster_centers[0, ] = np.copy(data[index, ])

def cv_show(img, name='img'):
    '''
    显示图片
    :param img: 图片
    :param name: 标题
    :return:
    '''
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def kmeans(data, n, h, w):
    '''
    kmeans
    :param data: 样本
    :param n: 聚类个数
    :param h: 图片高
    :param w: 图片宽
    :return:
    '''
    start = time.time()
    # 初始化质心
    # K = np.random.uniform(0, 1, (n, data.shape[1]))
    # K[0] = [1, 1, 1]
    # K[1] = [1, 1, 1]
    # K[2] = [1, 1, 1]

    # num = data.shape[0]
    # K = np.array([
    #     data[np.random.randint(0, num)],
    #     data[np.random.randint(0, num)],
    #     data[np.random.randint(0, num)]
    # ])
    # K = K.astype(float)
    K = np.random.uniform(0, 256, (n, data.shape[1]))
    print("初始化质心: {}".format(K))
    # 聚类结果 每个点对应一个数组
    ret = np.zeros((data.shape[0], 2))
    flag = True
    while flag:
        flag = False
        for i in range(data.shape[0]):
            # print("第{}个点: {}".format(i, data[i]))
            minDist = np.inf
            minIndex = -1
            # 遍历每个质心点
            for j in range(K.shape[0]):
                # 计算每个像素点到质心点的距离
                dst = distance(data[i], K[j])
                # print("第{}个质心点: {}, 距离: {}".format(j, K[j], dst))
                if dst < minDist:
                    minDist = dst
                    minIndex = j

            # 更新聚类结果，第一个元素是点位距离质心点的距离，第二个元素是点位属于哪个簇
            ret[i][0] = minDist
            ret[i][1] = minIndex
            # print("第{}个元素属于{}簇".format(i, minIndex))
        # print(ret)
        # 对每个簇，计算簇中所有点的均值，并作为新的质心
        for i in range(n):
            cluster = data[ret[:, 1] == i]
            if len(cluster) != 0:
                # 求所有点RGB均值，返回一个RGB值
                center = np.mean(cluster, axis=0)
                # print("簇中心: {}".format(center))
                # print("K[i]: {}".format(K[i]))
                if (abs(center - K[i]) <= 1).all():
                    pass
                else:
                    # 有质心改变，继续循环
                    flag = True
                    K[i] = center
    end = time.time()
    print("kmeans所用时间：{}".format(end - start))
    # 质心不再改变
    print("质心点为:\n{}".format(K))
    # n个簇的点位数
    # for i in range(n):
    #     print("{}簇有{}个点".format(i, len(data[ret[:, 1] == i])))
    # print("data",data)
    # print("ret",ret)
    label = np.transpose(ret[:, 1].reshape([h, w]))
    # print("label", label)
    # pic_new = image.new("L", (w, h))
    pic_part = image.new("RGB", (w, h))
    color = [(0, 0, 255, 255),
             (0, 255, 255, 255),
             (255, 255, 0, 255)]  # 蓝青黄
    index = 0
    centers = K
    centersTmp = copy.copy(centers)
    centersMap = {}
    for i in range(n):
        for j in range(i+1, n):
            if centersTmp[i][index] > centersTmp[j][index]:
                centersTmp[i], centersTmp[j] = copy.copy(centersTmp[j]), copy.copy(centersTmp[i])
    for i in range(n):
        centersMap[centersTmp[i][index]] = i

    for i in range(w):
        for j in range(h):
            # pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
            # pic_part.putpixel((i, j), color[int(label[i][j])])
            # print(centers[label[i][j]][index])
            pic_part.putpixel((i, j), color[centersMap[centers[int(label[i][j])][index]]])
    # pic_new.save("gray.jpg", "JPEG")
    pic_part.save("rgb.jpg", "JPEG")

if __name__ == '__main__':
    img, h, w = loadPicture("2.png")
    kmeans(img, 3, h, w)
