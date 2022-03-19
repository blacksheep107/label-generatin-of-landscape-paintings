# label-generatin-of-landscape-paintings
Semantic segmentation and label generation of landscape paintings

- 本科毕业设计

# 风景画作语义分割及标签图生成

## 使用说明

- python3.6.15+ django + opencv，具体环境配置查看plist.txt，安装方式：
```
pip install -r plist.txt
```
实际上项目中很多包是不需要装的，只要注意包版本就行，有的包版本不同会有冲突，还有的安装报错是虚拟环境下要用conda install等等。这个列表好像是直接从虚拟环境导入的，前期尝试的方法很多，有很多最后没采用。
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

## 部署步骤和坑
- 参考的是菜鸟教程的Django教程，最后一节有Django + Nginx + uwsgi的部署。
- Ubuntu默认python3是3.4的好像，百度找一下装个3.6.15的。然后在/home/.virtualenvs下开个python虚拟环境
```
python -m venv env
```
激活虚拟环境，pip装包都在虚拟环境的状态下进行，之后运行uwsgi也一定要在虚拟环境下做
```
source /home/.virtualenvs/env/bin/activate
```
好像还有virtualenvwrapper和virtualenv的安装，这两个装的过程中我一直报错，后来直接用python自带的venv命令了，也不知道这两个包到底有没有用。
- 在虚拟环境下pip install必要的python包，版本一定要对。
- 装好uwsgi，按菜鸟教程上的测一下uwsgi装成功了没有，再测一下django项目能不能跑
    - django在服务器上的测试方式和本地一样，都是python manage.py runserver 0.0.0.0:8001，然后就可以访问公网ip:8001，可以正常访问就说明django项目是没问题的，必须保证这个。
- 测好uwsgi和django都正常后，接下来要把nginx和django通过uwsgi连起来，这步报了很多问题。
    - 配置uswgi.ini文件和nginx.conf文件，注意uswgi的目录位置。
    - uwsgi.ini文件中的各个参数，chdir是项目在服务器中的目录位置，就是manage.py的位置，module就是 项目名.wsgi:application（一定要:application，这是一个坑），socket是和nginx通信的用的（这里设置的端口号和nginx.conf的uwsgi_pass一定要相同，但端口号要和nginx listen的区分开！），master启用process manager管理进程，vacuum退出时删文件，py-autoreload多长时间内触发重载，下面3个是设置最长响应时间的，放在nginx.conf里生效。
    - nginx.conf
        - server_name写localhost还是127.0.0.1还是公网ip也有讲究，这里用公网ip没问题，之前写localhost好像有ipv6解析的问题（报错502），但别的项目写localhost就没问题，不懂。
        - 算法花时间长，要改最长响应时间，主要是uwsgi_connet_read_timeout，这个对应的是proxy_read_timeout。
        - location /下面，uwsgi_pass一定要和uwsgi.ini里的一样，include uwsgi_params直接写绝对路径，这是个文件
- 设置好了访问ip:82端口，就是listen的那个端口。改nginx.conf要重启nginx，改uwsgi.ini要重启uwsgi.ini（只有这一个python项目，直接killall -9 uwsgi.ini，然后再在项目路径下uwsgi --ini uwsgi.ini）。
- 之后就可以正常访问了。
- uwsgi默认是在命令行里运行，后面加个&是在后台运行
```
uwsgi --ini uwsgi.ini&
```
- 报错502可能：
    - 两个文件没配置好。
- 报错500可能：
    - 没在虚拟环境下运行。这个是报internet server error
    - uwsgi没用**不挂起后台运行**，用命令行运行完就直接退出了，这时候只能访问模板页面，但运行算法就会报Server Error (500)。在命令行启动uwsgi时可以正常访问，退出后就不行了，应该：
    ```
    nohup /home/.virtualenvs/gasyori100/bin/uwsgi --ini /home/djangopro/uwsgi.ini&
    ```
    nohup是 no hang up（不挂起），用于在系统后台不挂断地运行命令，退出终端不会影响程序的运行，&是让命令在后台执行，终端退出后命令仍旧执行。这两个都要有！！！```/home/.virtualenvs/gasyori100/bin/```指定虚拟环境，```/home/djangopro/uwsgi.ini```指定uwsgi.ini文件位置。
    - 程序报错了。删掉static下面的图片试试看（不一定有用）估计是kmeans的多线程出问题（不是）。这个点是因为手机和电脑访问状况不同找出来的，电脑访问有时是正常的有时502，并且跟输入图片大小有关系（因此误以为是内存或者CPU不够用）；手机访问每一次都是502，小图片也是，后来发现是uwsgi.ini参数。
    ```
        http-timeout: 3600
        socket-timeout: 3600
        buffer-size: 10240
        harakiri:3600
    ```
    - http-timeout和socket-timeout是连接时间，比如服务器需要5分钟才能响应，这两个值的时间就要大于5分钟，否则1分钟后就断开网络连接了，但服务器还是会把程序运行完，可这是前段已经显示502了。
    - buffer-size是传输数据的大小，肯定是要比图片大的，这里直接设置10M
    - harakiri，服务器运行程序的限制时间，比如这个程序要跑5分钟，但harakiri只有10s，那么10s后服务器就强制终止计算了，所以一定要设置大于5min。
- 报错504可能：
    - 没改时间，f12看包time，1min就是时间没改好。 