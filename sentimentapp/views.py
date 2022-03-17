from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
import requests
import numpy as np
import pandas as pd
import os
from requests.api import get
import pandas as pd
import numpy as np
import requests
import pickle
from numpy import array
from pandas.io.json import json_normalize
from docx import Document
import docx
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

#this url for accessing api
URL = "http://127.0.0.1:8002/sentimentapi/"

#this file for model
path = os.path.join(os.path.dirname(__file__), 'models.p')

#read the pickle file here
with open(path, 'rb') as pickled:
        data = pickle.load(pickled)

#store model and vectorizer
model = data['model']
vectorizer = data['vectorizer']

#this function convert the file data to text
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

#This is for homepage
def home(request):
    return render(request, 'home.html')

#This is main function where we done everything for sentiment analysis
def sentiment(request):
    try:
        if request.method == 'POST':
            #this segment for working with file
            if request.FILES:
                rawtext = request.FILES['filename']
                x = getText(rawtext)
                # predict the sentiment here
                vector = vectorizer.transform([x])
                prediction = model.predict(vector)[0]
                mydict = {
                    "mytext" : x,
                    "sentiment" : prediction,
                }
                json_data = json.dumps(mydict)
                # request api for post the data
                r = requests.post(url = URL, data=json_data)
                data = r.json()
            else:
                #this segment for working with text
                text = request.POST['rawtext']
                # predict the sentiment here
                vector = vectorizer.transform([text])
                prediction = model.predict(vector)[0]
                mydict = {
                    "mytext" : text,
                    "sentiment" : prediction,
                }
                json_data = json.dumps(mydict)
                # request api for post the data
                r = requests.post(url = URL, data=json_data)
                data = r.json()

        return render(request, 'home.html', context=mydict)

    except: 
        return HttpResponse("Opps...!\nSomething else went wrong.")  