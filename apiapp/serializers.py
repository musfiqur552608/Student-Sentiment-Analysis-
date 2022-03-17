from rest_framework import serializers
from .models import Sentiment

class SentimentSerializer(serializers.Serializer):
    mytext = serializers.CharField(max_length=1000000)
    sentiment = serializers.CharField(max_length=100)
    
    # created this function for create, read, delete
    def create(self, validated_data):
        return Sentiment.objects.create(**validated_data)
    
    # created this function for update
    def update(self, instance, validated_data):
        instance.mytext = validated_data.get('mytext', instance.mytext)
        instance.sentiment = validated_data.get('sentiment', instance.sentiment)
        instance.save()
        return instance