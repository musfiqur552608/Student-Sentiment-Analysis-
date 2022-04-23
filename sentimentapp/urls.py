from os import name
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='index'),
    path('sentiment', views.sentiment, name='sentiment'),
    path('train', views.train, name='train'),
]