import numpy as np
import sys
from skimage import io, color
import tqdm
import cv2
# using algorithm in 3.2 apply image gradients as computed in eq2:
# G(x,y) = ||I(x+1,y) - I(x-1,y)||^2+ ||I(x,y+1) - I(x,y-1)||^2

# SLIC implements a special case of k-means clustering algorithm.
# Was recommended to use an off the shelf algorithm for clustering but
# because this algorithm is based on this special case of k-means,
# I kept this implementation to stay true to the algorithm.

def generate_pixels():
    '''
    生成超像素
    :return:
    '''
    # mgrid生成height*width的矩阵，把0,2轴交换，然后0,1轴交换，结果是00,01,02...10,11,12的一个遍历矩阵
    indnp = np.mgrid[0:SLIC_height,0:SLIC_width].swapaxes(0,2).swapaxes(0,1)
    # 进度条
    for i in tqdm.tqdm(range(SLIC_ITERATIONS)):
        # 距离数组，每个像素和对应的簇中心的距离
        SLIC_distances = 1 * np.ones(img.shape[:2])
        # 遍历簇中心
        for j in range(SLIC_centers.shape[0]):
            # 一个超像素的长宽边界值
            x_low, x_high = int(SLIC_centers[j][3] - step), int(SLIC_centers[j][3] + step)
            y_low, y_high = int(SLIC_centers[j][4] - step), int(SLIC_centers[j][4] + step)

            if x_low <= 0:
                x_low = 0
            #end
            if x_high > SLIC_width:
                x_high = SLIC_width
            #end
            if y_low <=0:
                y_low = 0
            #end
            if y_high > SLIC_height:
                y_high = SLIC_height
            #end

            # 范围内所有像素点距离中心点的颜色距离（欧氏距离）
            cropimg = SLIC_labimg[y_low : y_high , x_low : x_high]
            color_diff = cropimg - SLIC_labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])]
            color_distance = np.sqrt(np.sum(np.square(color_diff), axis=2))
            # 范围内所有像素点距离中心点的空间距离（欧式距离）
            yy, xx = np.ogrid[y_low : y_high, x_low : x_high]
            pixdist = ((yy-SLIC_centers[j][4])**2 + (xx-SLIC_centers[j][3])**2)**0.5

            # SLIC_m is "m" in the paper, (m/S)*dxy
            # 5维距离，按论文中的简化公式计算
            dist = ((color_distance/SLIC_m)**2 + (pixdist/step)**2)**0.5
            distance_crop = SLIC_distances[y_low : y_high, x_low : x_high]
            # idx是二维数组，distance_crop中的每个值是否大于dist
            idx = dist < distance_crop
            # 大于dist的重新赋值
            distance_crop[idx] = dist[idx]
            SLIC_distances[y_low : y_high, x_low : x_high] = distance_crop
            SLIC_clusters[y_low : y_high, x_low : x_high][idx] = j
        #end

        # 调整聚类中心
        for k in range(len(SLIC_centers)):
            idx = (SLIC_clusters == k)
            colornp = SLIC_labimg[idx]
            distnp = indnp[idx]
            SLIC_centers[k][0:3] = np.sum(colornp, axis=0)
            sumy, sumx = np.sum(distnp, axis=0)
            SLIC_centers[k][3:] = sumx, sumy
            # print(SLIC_centers[k])
            # print(np.sum(idx))
            # print(SLIC_centers[k] / np.sum(idx))
            SLIC_centers[k] = SLIC_centers[k] / np.sum(idx)
            # SLIC_centers[k] /= np.sum(idx)
        #end
    #end
#end

def display_contours(color):
    rgb_img = img.copy()
    is_taken = np.zeros(img.shape[:2], np.bool)
    contours = []

    for i in range(SLIC_width):
        for j in range(SLIC_height):
            nr_p = 0
            for dx, dy in [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1)]:
                x = i + dx
                y = j + dy
                if x>=0 and x < SLIC_width and y>=0 and y < SLIC_height:
                    if is_taken[y, x] == False and SLIC_clusters[j, i] != SLIC_clusters[y, x]:
                        nr_p += 1
                    #end
                #end
            #end

            if nr_p >= 2:
                is_taken[j, i] = True
                contours.append([j, i])
            #end
        #end
    #end
    for i in range(len(contours)):
        rgb_img[contours[i][0], contours[i][1]] = color
    # for k in range(SLIC_centers.shape[0]):
    #     i,j = SLIC_centers[k][-2:]
    #     img[int(i),int(j)] = (0,0,0)
    #end
    # io.imsave("SLIC_contours.jpg", rgb_img)
    cv2.imwrite("SLIC_contours.jpg", rgb_img)
    return rgb_img
#end

def display_center():
    '''
    将超像素用聚类中心颜色代替
    '''

    lab_img = np.zeros([SLIC_height,SLIC_width,3]).astype(np.float32)
    for i in range(SLIC_width):
        for j in range(SLIC_height):
            k = int(SLIC_clusters[j, i])
            lab_img[j,i] = SLIC_centers[k][0:3]
    rgb_img = color.lab2rgb(lab_img)
    rgb_img = (rgb_img*255).astype(np.uint8)
    cv2.imwrite("SLIC_centers.jpg", rgb_img)
    return rgb_img

def find_local_minimum(center):
    '''
    计算3*3内梯度最小的点
    :param center: 中心点
    :return: 梯度最小的点
    '''
    min_grad = 1
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = SLIC_labimg[j+1, i]
            c2 = SLIC_labimg[j, i+1]
            c3 = SLIC_labimg[j, i]
            # if ((c1[0] - c3[0])**2)**0.5 + ((c2[0] - c3[0])**2)**0.5 < min_grad:
            tmp = abs(int(c1[0]) - int(c3[0])) + abs(int(c2[0]) - int(c3[0]))
            if tmp < min_grad:
                min_grad = tmp
                loc_min = [i, j]
            #end
        #end
    #end
    return loc_min
#end

def calculate_centers():
    '''
    计算簇中心
    :return: 簇中心点数组，每个元素5维，labxy
    '''
    centers = []
    for i in range(step, SLIC_width - int(step/2), step):
        for j in range(step, SLIC_height - int(step/2), step):
            nc = find_local_minimum(center=(i, j))  # 3*3内梯度最小的点
            color = SLIC_labimg[nc[1], nc[0]]
            center = [color[0], color[1], color[2], nc[0], nc[1]]
            centers.append(center)
        #end
    #end

    return centers
#end
def cv_show(img, name='img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("2.png")
# 中值滤波
img = cv2.medianBlur(img, 5)
# print(img.max(),img.min())
# step = int((img.shape[0]*img.shape[1]/int(sys.argv[2]))**0.5)
# 初始分割块数
k = 500
# k = 1000
# 给定集群中预期的最大空间距离
step = int((img.shape[0]*img.shape[1]/int(k))**0.5)
# SLIC_m = int(sys.argv[3])
# m越大空间邻近度越重要，m越小超像素会更紧密地粘附在颜色边缘
SLIC_m = int(10)
# 迭代次数
SLIC_ITERATIONS = 10
SLIC_height, SLIC_width = img.shape[:2]
# RGB转LAB
SLIC_labimg = color.rgb2lab(img)
# SLIC_labimg = img
# 每个像素对超像素中心的距离
SLIC_distances = 1 * np.ones(img.shape[:2])
# 每个像素点属于哪个簇
SLIC_clusters = -1 * SLIC_distances
centers = calculate_centers()
# 中心点数
SLIC_center_counts = np.zeros(len(centers))
# 中心点数组
SLIC_centers = np.array(centers)

# main
generate_pixels()
img_contours = display_contours([0.0, 0.0, 0.0])
img_center = display_center()

# print(img,img_center,img_contours)
result = np.hstack([img,img_contours,img_center])
cv2.imwrite("my_slic.jpg", result)
# io.imsave("my_slic.jpg",result)   # cv2是BGR顺序，用io存储会出现颜色相反
