from django.db import models

# Create your models here.
class Sentiment(models.Model):
    mytext = models.CharField(max_length=1000000)
    sentiment = models.CharField(max_length=100)