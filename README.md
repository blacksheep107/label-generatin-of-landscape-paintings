# label-generatin-of-landscape-paintings
Semantic segmentation and label generation of landscape paintings

- 本科毕业设计

# 风景画作语义分割及标签图生成

## 使用说明

- python3.6.15+ django + opencv，具体环境配置查看plist.txt，安装方式：
```
pip install -r plist.txt
```
实际上项目中很多包是不需要装的，这个列表好像是直接从虚拟环境导入的，前期尝试的方法很多，有很多最后没采用。
- 前端用django模板写，在本地运行
```
python manage.py runserver 0.0.0.0:8000
```
访问0.0.0.0:8000操作图形化界面。
- 可部署，尝试在阿里云服务器（Ubuntu）部署，用uWSGI + nginx做服务，可以成功运行。
- 算法运行时间很长，图像大的话一分钟往上，部署的话需要设置nginx最长等待时间，否则会504
- 部署的坑非常多，特别是uwsgi.ini和nginx.conf的参数上。

## 语义分割步骤

- kmeans做第一层分割，将风景画分为云水（黄）、山石（蓝）、植被（青）3簇。
- 提取图像纹理，用Gabor + GLCM灰度共生矩阵，生成熵纹理特征影像，提取出植被区域纹理，二值化生成遮罩。
- SLIC超像素分割处理原图像，转HSV颜色空间方便提取绿色。
- 结合遮罩和HSV绿色范围提取植被。