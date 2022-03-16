from django.http import HttpResponse
from django.shortcuts import render
import time
def serch_from(request):
    # return render(request, 'hello.html')
    return HttpResponse("hello?")
def search(request):
    request.encoding = 'utf-8'
    if 'q' in request.GET and request.GET['q']:
        message = '你搜索的内容为：' + request.GET['q']
    else:
        message = '你提交了空表单'
    return HttpResponse(message)