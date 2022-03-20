from django.http import HttpResponse
from django.shortcuts import render
import os
import base64
import cv2
import numpy as np
from . import kmeans
from PIL import Image
def hello(request):
    if request.method == 'GET':
         # return HttpResponse(1)
        return render(request, 'test.html')
        # return HttpResponse(file.read(), content_type="image/png")
    elif request.method == 'POST':
        imgfile = request.FILES.get('img')
        img = Image.open(imgfile)
        postObj = request.POST
        print(postObj)
        print(os.path.join(os.getcwd(), 'static/images', "img.png"))
        img.save(os.path.join(os.getcwd(), 'static/images', "img.png"))
        print('save ok')
        inertia = kmeans.main(os.path.join(os.getcwd(), 'static/images', "img.png"),
                    postObj['type'],
                    postObj['step'],
                    postObj['angle'],
                    postObj['slic_num'],
                    postObj['compactness_num'],
                    postObj['gabor_fre'])
        context = {
            'origin_img': False,
            'noplant_img': False,
            'filt_img': False,
            'GLCM_Entropy': False,
            'slic_result': False,
            'hsvMask': False,
            'result': False,
            'type': postObj['type'],
            'slic_num': postObj['slic_num'],
            'compactness_num': postObj['compactness_num'],
            'step': postObj['step'],
            'angle': postObj['angle'],
            'gabor_fre': postObj['gabor_fre'],
            'inertia': inertia
        }
        contextArr = ['origin_img', 'noplant_img', 'filt_img', 'GLCM_Entropy','slic_result', 'hsvMask', 'result']
        pathArr = [
            'img.png', 'noplant.png', 'filt_imag.png', 'GLCM_Features.png','slic_result.png', 'hsvMask.png', 'result.png'
        ]
        for i in range(len(pathArr)):
            path = os.path.join(os.getcwd(), 'static/images', pathArr[i])
            img_stream = ''
            if os.path.exists(path):
                with open(path, 'rb') as img_f:
                    img_stream = img_f.read()
                    img_stream = base64.b64encode(img_stream)
                    context[contextArr[i]] = "data:image/jpg;base64," + str(img_stream)[2: -1]
        return render(request, 'test.html', context)
