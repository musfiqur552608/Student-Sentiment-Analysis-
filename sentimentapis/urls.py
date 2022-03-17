from django.contrib import admin
from django.urls import path, include
from apiapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('sentimentapp.urls')),
    path('sentimentapi/', views.SentimentApi.as_view())
]