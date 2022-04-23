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
##############################################
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
##############################################

#this url for accessing api
URL = "http://127.0.0.1:8002/sentimentapi/"

#this file for model
path = os.path.join(os.path.dirname(__file__), '..\models.p')

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


wordnet_lemmatizer = WordNetLemmatizer()
def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
    only_letters = only_letters.lower()
    only_letters = only_letters.split()
    filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    lemmas = ' '.join(lemmas)
    return lemmas
def train(request):
    # try:
        if request.method == 'POST':
            #this segment for working with file
            if request.FILES:
                train = request.FILES['train']

                df = pd.read_csv(train)
                nltk.download('wordnet')
                df = shuffle(df)
                y = df['airline_sentiment']
                x = df.text.apply(normalizer)

                vectorizer = CountVectorizer()
                x_vectorized = vectorizer.fit_transform(x)

                train_x,val_x,train_y,val_y = train_test_split(x_vectorized,y)

                regressor = LogisticRegression(multi_class='multinomial', solver='newton-cg')
                model = regressor.fit(train_x, train_y)

                params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
                gs_clf = GridSearchCV(model, params, n_jobs=1, cv=5)
                gs_clf = gs_clf.fit(train_x, train_y)
                model = gs_clf.best_estimator_

                y_pred = model.predict(val_x)

                _f1 = f1_score(val_y, y_pred, average='micro')
                _confusion = confusion_matrix(val_y, y_pred)
                __precision = precision_score(val_y, y_pred, average='micro')
                _recall = recall_score(val_y, y_pred, average='micro')
                _statistics = {'f1_score': _f1,
                            'confusion_matrix': _confusion,
                            'precision': __precision,
                            'recall': _recall
                            }

                print(y_pred)
                print(_statistics)

                pickl = {'vectorizer': vectorizer,
                        'model': model
                        }
                pickle.dump(pickl, open('models'+".p", "wb"))
            

        return render(request, 'home.html')

    # except: 
    #     return HttpResponse("Opps...!\nSomething else went wrong.")  


