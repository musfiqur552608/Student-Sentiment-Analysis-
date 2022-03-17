from functools import partial
import json
from django.shortcuts import render
import io
from rest_framework import serializers

from rest_framework.parsers import JSONParser
from .models import Sentiment
from .serializers import SentimentSerializer
from rest_framework.renderers import JSONRenderer
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.utils.decorators import method_decorator
from django.views import View

#This class will control our apis
@method_decorator(csrf_exempt, name='dispatch')
class SentimentApi(View):
    #This function will use for get something via api
    def get(self, request, *args, **kwargs):
        json_data = request.body
        stream = io.BytesIO(json_data) 
        pythondata = JSONParser().parse(stream)
        id = pythondata.get('id', None)
        if id is not None:
            stu = Sentiment.objects.get(id=id)
            serializer = SentimentSerializer(stu)
            return JsonResponse(serializer.data, safe=False)
        stu = Sentiment.objects.all()
        serializer = SentimentSerializer(stu, many=True)
        return JsonResponse(serializer.data, safe=False)

    #This function will use for send something via api
    def post(self, request, *args, **kwargs):
        json_data = request.body
        stream = io.BytesIO(json_data)
        pythondata = JSONParser().parse(stream)
        serializer = SentimentSerializer(data=pythondata)
        if serializer.is_valid():
            serializer.save()
            res = {'msg' : 'Data Created'}
            return JsonResponse(res, safe=False)
        return JsonResponse(serializer.errors, safe=False)

    #This function will use for update something via api
    def put(self, request, *args, **kwargs):
        json_data = request.body
        stream = io.BytesIO(json_data)
        pythondata = JSONParser().parse(stream)
        id = pythondata.get('id')
        stu = Sentiment.objects.get(id=id)
        serializer = SentimentSerializer(stu, data = pythondata, partial=True)
        if serializer.is_valid():
            serializer.save()
            res = {'msg':'Data Updated !!'}
            return JsonResponse(res, safe=False)
        return JsonResponse(serializer.errors, safe=False)

    #This function will use for delete something via api
    def delete(self, request, *args, **kwargs):
        json_data = request.body
        stream = io.BytesIO(json_data)
        pythondata = JSONParser().parse(stream)
        id = pythondata.get('id')
        stu = Sentiment.objects.get(id=id)
        stu.delete()
        res = {'msg': 'Data Deleted!!'}
        # json_data = JSONRenderer().render(res)
        # return HttpResponse(json_data, content_type='application/json')
        return JsonResponse(res, safe=False)