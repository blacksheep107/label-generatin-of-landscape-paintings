import io

import numpy as np
import PIL.Image as image
import os
from skimage import filters
import cv2
from django.http import HttpResponse, Http404, FileResponse
import copy
from skimage import data
from matplotlib import pyplot as plt
from . import get_glcm
import time
from sklearn.cluster import KMeans
from PIL import Image
import base64
from . import SLIC

def loadData(filePath, type):
    '''
    加载图像
    :param filePath: 图像路径
    :return: 处理后的图像RGB二维数组
    '''
    print(type)
    if not os.path.exists(filePath):
        exit(-1)
    img = cv2.imread(filePath)
    h, w, z = np.shape(img)
    # 中值滤波
    img_median = cv2.medianBlur(img, 5)
    if type == 'gray':
        img_median = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)
        img_median = img_median.reshape(-1, 1)
        return img_median, h, w

    elif type == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif type == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # img = cv2.resize(img, (int(w / 4), int(h / 4)))
    print(h, w)
    h, w, z = np.shape(img)
    data = []
    for i in range(h):
        for j in range(w):
            # 归一化
            # data.append([img_median[i][j][0] / 256.0, img_median[i][j][1] / 256.0, img_median[i][j][2] / 256.0])
            data.append([img_median[i][j][0], img_median[i][j][1], img_median[i][j][2]])
    return np.array(data), h, w

def cv_show(img, name='img'):
    '''
    展示图像
    :param img: 图像
    :param name: 标题
    :return:
    '''
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def kmeans(n, randomState, data, type):
    '''
    kmeans
    :param n: 簇个数
    :param randomState: 随机状态，用来复现
    :param data: 聚类数据
    :return: labels每个数据的标签，centers簇中心
    '''
    if type == 'HSV':
        # 只用value聚类
        data = data[:,2].reshape(-1, 1)
    # elif type == 'RGB': # BGR
    #     data = data[:, 2].reshape(-1, 1)
    kmeans = KMeans(n_clusters=n, random_state=randomState).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_

def getHSV(filename, slic_num, compactness_num):
    '''
    slic图像转HSV颜色模型
    :param filename: 图片路径
    :return: HSV二维数组
    '''
    imgSLIC = SLIC.main(filename, slic_num, compactness_num)
    imgHSV = cv2.cvtColor(imgSLIC, cv2.COLOR_BGR2HSV)
    imgNoSLICHSV = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(os.getcwd(), 'static/images', "hsvMask.png"), imgHSV)
    # cv2.imwrite(os.path.join(os.getcwd(), 'static\\images', "hsvNoSLICMask.png"), imgNoSLICHSV)
    return imgHSV, imgNoSLICHSV

def gaborGetTexture(filename, gabor_fre=0.6):
    '''
    gabor滤波器提取纹理
    :return: 纹理图
    '''
    img = cv2.imread(filename, 0)
    img = cv2.medianBlur(img, 5)
    filt_real, filt_imag = filters.gabor(img, frequency=gabor_fre)
    cv2.imwrite(os.path.join(os.getcwd(), 'static/images', "filt_imag.png"), filt_imag)
    return filt_imag

def getGLCM(filename, st, an, gabor_fre):
    start = time.time()
    img = gaborGetTexture(filename, gabor_fre)
    print('---------------0. Parameter Setting-----------------')
    nbit = 16 # gray levels
    mi, ma = 0, 255 # max gray and min gray
    # slide_window = 31 # sliding window
    slide_window = 19 # sliding window
    # step = [2, 4, 8, 16] # step
    angleArr = [0, np.pi/4, np.pi/2, np.pi*3/4] # angle or direction
    step = [int(st)]
    angle = [angleArr[int(an)]]
    print('-------------------1. Load Data---------------------')
    # image = r"./test.tif"
    # img = cv2.imread("filt_imag.jpg", 0)
    # img = cv2.imread("2.png", 0)
    # img = np.array(Image.open(image)) # If the image has multi-bands, it needs to be converted to grayscale image
    img = np.uint8(255.0 * (img - np.min(img))/(np.max(img) - np.min(img))) # normalization
    print(img)
    h, w = img.shape
    print('------------------2. Calcu GLCM---------------------')
    glcm, size = get_glcm.calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)
    print('-----------------3. Calcu Feature-------------------')
    #
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit, nbit, h, w), dtype=np.float32)
            glcm_cut = glcm[:, :, i, j, :, :]
            # mean = get_glcm.calcu_glcm_mean(glcm_cut, nbit)
            # variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
            # homogeneity = get_glcm.calcu_glcm_homogeneity(glcm_cut, nbit)
            # contrast = get_glcm.calcu_glcm_contrast(glcm_cut, nbit)
            # dissimilarity = get_glcm.calcu_glcm_dissimilarity(glcm_cut, nbit)
            entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)
            # energy = get_glcm.calcu_glcm_energy(glcm_cut, nbit)
            # correlation = get_glcm.calcu_glcm_correlation(glcm_cut, nbit)
            # Auto_correlation = get_glcm.calcu_glcm_Auto_correlation(glcm_cut, nbit)
    print('---------------4. Display and Result----------------')
    # cv2.imwrite(os.path.join(os.getcwd(), 'static\\images', "test.png"), cv2.cvtColor(entropy, cv2.COLOR_GRAY2BGR))
    plt.figure()
    # font = {'family' : 'Times New Roman',
    # 'weight' : 'normal',
    # 'size'   : 12,
    # }
    # cv2.imshow("", entropy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.subplot(2,5,1)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(img, cmap ='gray')
    # plt.title('Original', font)
    #
    # plt.subplot(2,5,2)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(mean, cmap ='gray')
    # plt.title('Mean', font)
    #
    # plt.subplot(2,5,3)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(variance, cmap ='gray')
    # plt.title('Variance', font)
    #
    # plt.subplot(2,5,4)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(homogeneity, cmap ='gray')
    # plt.title('Homogeneity', font)
    #
    # plt.subplot(2,5,5)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(contrast, cmap ='gray')
    # plt.title('Contrast', font)
    #
    # plt.subplot(2,5,6)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(dissimilarity, cmap ='gray')
    # plt.title('Dissimilarity', font)
    #
    # plt.subplot(2,5,7)
    # # plt.subplot(2,5,2)
    # entropy = entropy[:, size:len(entropy)]
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(entropy, cmap ='gray')
    # plt.title('Entropy', font)
    #
    # plt.subplot(2,5,8)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(energy, cmap ='gray')
    # plt.title('Energy', font)
    #
    # plt.subplot(2,5,9)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(correlation, cmap ='gray')
    # plt.title('Correlation', font)
    #
    # plt.subplot(2,5,10)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.axis('off')
    # plt.imshow(Auto_correlation, cmap ='gray')
    # plt.title('Auto Correlation', font)

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(os.getcwd(), 'static/images', "GLCM_Features.png")
                , format='png'
                , bbox_inches = 'tight'
                , pad_inches = 0
                , dpi=300)
    # plt.show()

    end = time.time()
    print('GLCM run time:', end - start)

def getMask(filename, row, col):
    mask = cv2.imread(filename, 0)
    mask= cv2.resize(mask, (col, row), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(os.getcwd(), 'static/images', "GLCM_Features.png"), mask)
    ret, maskThresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return maskThresh

def getResult(centers, row, col, maskThresh, n, label, imgHSV, type, imgNoSLICHSV):
    '''
    生成图片
    :param centers: 簇中心
    :param row:
    :param col:
    :param maskThresh: 二值化遮罩图片
    :return:
    '''
    pic_part = image.new("RGBA", (row, col))
    pic_new = image.new("RGBA", (row, col))
    pic_noslic = image.new("RGBA", (row, col))
    # pic_notexture = image.new("RGBA", (row, col))
    color = [(0, 0, 255, 255),
             (0, 255, 255, 255),
             (255, 255, 0, 255),
             (255, 0, 0, 255)]  # 蓝青黄红

    index = 0
    # if type == 'RGB':
    #     index = 0
    # elif type == 'HSV':
    #     index = 0
    #     # color[0], color[2] = color[2], color[0]
    # elif type == 'LAB':
    #     index = 0
    # 标签颜色固定
    centersTmp = copy.copy(centers)
    centersMap = {}
    for i in range(n):
        for j in range(i+1, n):
            if centersTmp[i][index] > centersTmp[j][index]:
                centersTmp[i], centersTmp[j] = copy.copy(centersTmp[j]), copy.copy(centersTmp[i])
    for i in range(n):
        centersMap[centersTmp[i][index]] = i
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), color[centersMap[centers[label[i][j]][index]]])
            # if color[centersMap[centers[label[i][j]][index]]] != (255, 255, 0, 255) and imgHSV[j][i][0] in range(26, 99) and imgHSV[j][i][1] in range(43, 255) and imgHSV[j][i][2] in range(46, 255):
            #     pic_notexture.putpixel((i, j), (0, 200, 0, 255)) # 绿色
            # else:
            #     pic_notexture.putpixel((i, j), color[centersMap[centers[label[i][j]][index]]])

            flag = maskThresh[j][i] == 0 and color[centersMap[centers[label[i][j]][index]]] != (255, 255, 0, 255)
            if flag and imgHSV[j][i][0] in range(26, 99) and imgHSV[j][i][1] in range(43, 255) and imgHSV[j][i][2] in range(46, 255):
                pic_part.putpixel((i, j), (0, 200, 0, 255)) # 绿色
            else:
                pic_part.putpixel((i, j), color[centersMap[centers[label[i][j]][index]]])
            if flag and imgNoSLICHSV[j][i][0] in range(26, 99) and imgNoSLICHSV[j][i][1] in range(43, 255) and imgNoSLICHSV[j][i][2] in range(46, 255):
                pic_noslic.putpixel((i, j), (0, 200, 0, 255)) # 绿色
            else:
                pic_noslic.putpixel((i, j), color[centersMap[centers[label[i][j]][index]]])

    pic_new.save(os.path.join(os.getcwd(), 'static/images', "noplant.png"), "PNG")
    pic_part.save(os.path.join(os.getcwd(), 'static/images', "result.png"), "PNG")
    pic_noslic.save(os.path.join(os.getcwd(), 'static/images', "result_noslic.png"), "PNG")
    # pic_notexture.save(os.path.join(os.getcwd(), 'static\\images', "result_notexture.png"), "PNG")
    return pic_part

n = 3
# filename = os.path.join(os.getcwd(), 'static\\images', "img.png")
# if not os.path.exists(filename):
#     raise Http404

def main(filename, t, step, angle, slic_num, compactness_num, gabor_fre):
    type = t
    imgData, row, col = loadData(filename, type)
    # 转HSV颜色模型
    imgHSV, imgNoSLICHSV = getHSV(filename, slic_num, compactness_num)

    # 获取灰度共生矩阵熵
    getGLCM(filename, step, angle, float(gabor_fre))
    # 获取纹理遮罩
    maskThresh = getMask(os.path.join(os.getcwd(), 'static/images', "GLCM_Features.png"), row, col)

    # kmeans聚类
    label, centers, inertia = kmeans(n, None, imgData, type)
    print(inertia)
    # 转置
    label = np.transpose(label.reshape([row, col]))
    row, col = col, row
    # 写入标签图
    getResult(centers, row, col, maskThresh, n, label, imgHSV, type, imgNoSLICHSV)
    return inertia
    # path = os.path.join(os.getcwd(), 'static\\images', "part.png")
    # img_stream = ''
    # with open(path, 'rb') as img_f:
    #     img_stream = img_f.read()
    #     img_stream = base64.b64encode(img_stream)
    # return "data:image/jpg;base64," + str(img_stream)[2: -1]

# main(os.path.join(os.getcwd(), 'static\\images', "img.png"))
# main("img.png")